/*
   Copyright 2010 Joseph Reisinger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <math.h>
#include <time.h>

#include <algorithm>


#include "gibbs-base.h"
#include "sample-crosscat-mm.h"

// the number of feature clusters
DEFINE_int32(M,
             1,
             "Number of feature clusters / data views.");

// the maximum number of clusters
DEFINE_int32(KMAX,
             -1,
             "Maximum number of clusters.");

// Basically controls whether and how we should do cross-cat on the features.
// Implemented using MH steps.
DEFINE_string(cross_cat_prior,
              "uniform",
              "Prior over how we do cross-cat. Defaults to uniform.");

// Smoother on clustering
DEFINE_double(mm_alpha,
              1.0,
              "Smoother on the cluster assignments.");

// Smoother on cross cat clustering
DEFINE_double(cc_xi,
              1.0,
              "Smoother on the cross-cat cluster assignments.");

// Number of feature moves to make
DEFINE_double(cc_feature_move_rate,
             0.01,
             "Percentage of features to move per Gibbs sweep.");

// File holding the data
DEFINE_string(mm_datafile,
              "",
              "Docify holding the data to be clustered.");

// If toggled, the first view will be constrained to a single cluster
DEFINE_bool(cc_include_noise_view,
            false,
            "Should the first view be confined to a single cluster.");

const string kDirichletProcess = "dirichlet-process";
const string kDirichletMixture = "dirichlet";
const string kUniformMixture = "uniform";

void CrossCatMM::batch_allocation() {
    LOG(INFO) << "initialize";

    _b.set_empty_key(kEmptyUnsignedKey);
    _m.set_empty_key(kEmptyUnsignedKey);

    // Set up the asymmetric dirichlet priors for each clustering and for the
    // cross-cat
    
    // xi is the smoother over view assignments
    for (int l = 0; l < FLAGS_M; l++) {
        _xi.push_back(FLAGS_cc_xi);
    }
    _xi_sum = FLAGS_M*FLAGS_cc_xi;

    // Keep track of the number of clusters in each view
    _current_component.resize(FLAGS_M);

    // Allocate the initial clusters
    for (int m = 0; m < FLAGS_M; m++) {
        clustering t;
        t.set_empty_key(kEmptyUnsignedKey);
        t.set_deleted_key(kDeletedUnsignedKey);
        _c.insert(pair<unsigned,clustering>(m, t));

        // Push back a single cluster; we'll allocate more potentially later
        if (FLAGS_cc_include_noise_view && m == 0) {
            _c[m].insert(pair<unsigned,CRP>(0, CRP()));
            _current_component[m] = 1;
        }

        _b.insert(pair<unsigned,CRP>(m, CRP()));
    }

    // Allocate the initial feature clusters / mappings; uniformly distribute
    // features over views
    for (int w = 0; w < _lV; w++) {
        _m[w] = sample_integer(FLAGS_M);
        _b[_m[w]].ndsum += 1;

        VLOG(1) << _word_id_to_name[w] << " assigned to view " << _m[w];
    }

    // Add the documents into the clustering
    _lD = 0;  // reset this to make the resample_posterior_z stuff below work correctly
    for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;  // = document number
        _lD += 1;

        cluster_map t;
        t.set_empty_key(kEmptyUnsignedKey);
        _z.insert(pair<unsigned,cluster_map>(d, t));
        // _z[d].set_empty_key(kEmptyUnsignedKey);
        
        // Allocate this doc in the collapsed vector
        collapsed_document temp;
        // temp.set_empty_key(kEmptyUnsignedKey);
        _DD.insert(pair<unsigned,collapsed_document>(d, temp));

        // Collapse the document and compute per-view feature marginals
        for (int n = 0; n < d_itr->second.size(); n++) {
            unsigned w = d_itr->second[n];
            _DD[d][w] += 1;

            // This will store the marginal information about the frequency of
            // w, so we can compute a ranked list of features for each one.
            _b[_m[w]].nw[w] += 1;
            _b[_m[w]].nwsum += 1;
        }

        // TODO: deallocate the original _D

        if (d == 0) {
            // set a random cluster assignment for this document
            for (int m = 0; m < FLAGS_M; m++) {
                _z[d][m] = sample_integer(_c[m].size());
                _c[m][_z[d][m]].ndsum += 1; // Can't use ADD b/c we need to maintain ndsum over all the views
            }

            // Initial level assignments
            for (int n = 0; n < d_itr->second.size(); n++) {
                unsigned w = d_itr->second[n];
                unsigned zdm = _z[d][_m[w]];

                // test the initialization of maps
                CHECK(_c[_m[w]][zdm].nw.find(w) != _c[_m[w]][zdm].nw.end()
                        || _c[_m[w]][zdm].nw[w] == 0);

                // Cant use ADD b/c we need to keep the ndsum consistent across the
                // views
                // _c[_m[w]][zdm].add(w, d);
                _c[_m[w]][zdm].nw[w] += 1;
                _c[_m[w]][zdm].nwsum += 1;
            }
        } else {
            // Resample
            for (int m = 0; m < FLAGS_M; m++) {
                resample_posterior_z_for(d, m, false);
            }
        }

        if (d % 1000 == 0 && d > 0) {
          LOG(INFO) << "Sorted " << d << " documents";
        }

    }
    // Cull the DP assignments if there are any unused initial clusters.
    for (int m = 0; m < FLAGS_M; m++) {
        for (clustering::iterator itr = _c[m].begin();
                itr != _c[m].end();
                itr++) {
            if (itr->second.ndsum == 0) {  // empty component
                _c[m].erase(itr);
            }
        }
    }
    
    CHECK_LE(FLAGS_cc_feature_move_rate, 1.0);
    CHECK_GE(FLAGS_cc_feature_move_rate, 0.0);
    LOG(INFO) << "performing " << FLAGS_cc_feature_move_rate*100 << "\% feature moves per iteration.";

    // Actually try and assign all the features at least once
    // LOG(INFO) << "reallocating 10\% features initially";
    // resample_posterior_m(0.1);

    _ll = compute_log_likelihood();
    
}

void CrossCatMM::resample_posterior_m(double percent) {
    _m_proposed = _m_failed = 0;
    for (int f = 0; f < _lV; f++) {
        if (sample_uniform() < percent) {
            resample_posterior_m_for(f);
        }
    }
    LOG(INFO) << _m_proposed-_m_failed << " / " << _m_proposed << " " 
        << StringPrintf("(%.3f%%)", 100 - _m_failed / (double)_m_proposed*100) << " feature-view moves.";
}

// Performs a single document's level assignment resample step
void CrossCatMM::resample_posterior_z_for(unsigned d, unsigned m, bool remove) {
    clustering& cm = _c[m];

    unsigned old_zdm = 0;
    
    unsigned total_removed_count = 0;
    for (collapsed_document::iterator d_itr = _DD[d].begin();
            d_itr != _DD[d].end();
            d_itr++) {
        unsigned w = d_itr->first;
        unsigned count = d_itr->second;
        if (_m[w] == m) {
            total_removed_count += count;
        }
    }

    if (remove) {
        old_zdm = _z[d][m];
        // Remove this document from this clustering
        for (collapsed_document::iterator d_itr = _DD[d].begin();
                d_itr != _DD[d].end();
                d_itr++) {
            unsigned w = d_itr->first;
            unsigned count = d_itr->second;
            if (_m[w] == m) {
                // Remove this document and word from the counts
                cm[old_zdm].nw[w] -= count;  // # of words in topic z equal to w
                CHECK_GE(cm[old_zdm].nw[w], 0);
            }
        }
        CHECK_GT(cm[old_zdm].ndsum, 0);

        cm[old_zdm].nwsum -= total_removed_count;  // # of words in topic z
        cm[old_zdm].ndsum -= 1;  // # of docs in topic

        CHECK_GE(cm[old_zdm].nwsum, 0);
        CHECK_LE(cm[old_zdm].ndsum, _lD);
    }

    // Compute the log likelihood of each cluster assignment given all the other
    // document-cluster assignments
    vector<pair<unsigned,double> > lp_z_d;
    for (clustering::iterator itr = cm.begin();
            itr != cm.end();
            itr++) {
        unsigned l = itr->first;

        double sum = 0;
        
        // First add in the prior over the clusters
        sum += log(cm[l].ndsum) - log(_lD - 1 + FLAGS_mm_alpha);

        // Add in the normalizer for the multinomial-dirichlet likelihood
        sum += gammaln(_eta_sum + cm[l].nwsum) - gammaln(_eta_sum + cm[l].nwsum + total_removed_count);

        // Now account for the likelihood of the data (marginal posterior of
        // DP-Mult)
        for (collapsed_document::iterator d_itr = _DD[d].begin();
                d_itr != _DD[d].end();
                d_itr++) {
            unsigned w = d_itr->first;
            unsigned count = d_itr->second;

            if (_m[w] == m) {
                sum += gammaln(_eta[w] + count + cm[l].nw[w]) - gammaln(_eta[w] + cm[l].nw[w]);
            }
        }
        lp_z_d.push_back(pair<unsigned,double>(l, sum));
    }


    // Add an additional new component if not in the noise view
    if (cm.size() < FLAGS_KMAX || FLAGS_KMAX==-1) {
        if (m != 0 || !FLAGS_cc_include_noise_view) {
            double sum = log(FLAGS_mm_alpha) - log(_lD - 1 + FLAGS_mm_alpha);
            // Add in the normalizer for the multinomial-dirichlet likelihood
            sum += gammaln(_eta_sum) - gammaln(_eta_sum + total_removed_count);
            for (collapsed_document::iterator d_itr = _DD[d].begin();
                    d_itr != _DD[d].end();
                    d_itr++) {
                unsigned w = d_itr->first;
                unsigned count = d_itr->second;
                if (_m[w] == m) {
                    sum += gammaln(_eta[w] + count) - gammaln(_eta[w]);
                }
            }
            lp_z_d.push_back(pair<unsigned,double>(_current_component[m], sum));
        }
    }

    // Update the assignment
    _z[d][m] = sample_unnormalized_log_multinomial(&lp_z_d);
    VLOG(1) << "resampling posterior z for " << d << "," << m << ": " << old_zdm << "->" << _z[d][m];

    unsigned new_zdm = _z[d][m];

    if (new_zdm == old_zdm) {
        _c_failed += 1;
    }
    _c_proposed += 1;

    // Update the counts
    for (collapsed_document::iterator d_itr = _DD[d].begin();
            d_itr != _DD[d].end();
            d_itr++) {
        unsigned w = d_itr->first;
        unsigned count = d_itr->second;

        if (_m[w] == m) {
            cm[new_zdm].nw[w] += count;  // number of words in topic z equal to w
        }
    }
    cm[new_zdm].nwsum += total_removed_count;  // number of words in topic z
    cm[new_zdm].ndsum += 1;  // number of words in doc d with topic z

    CHECK_LE(cm[new_zdm].ndsum, _lD);

    // Clean up for the DPCrossCatMM
    if (cm[old_zdm].ndsum == 0) {  // empty component
        // LOG(INFO) << "removing cluster " << old_zdm << " from view "  << m << " because we chose cluster " << new_zdm;
        // _c.erase(old_zdm);
        _c[m].erase(old_zdm);
    }
    // Make room for a new component if we selected the new one
    if (new_zdm == _current_component[m]) {
        _current_component[m] += 1;
    }
}

// Likelihood of a particular cross-cat clustering of a particular word
double CrossCatMM::cross_cat_clustering_log_likelihood(unsigned w, unsigned m) {
    double log_lik = 0;
    
    // Likelihood of this particular feature w belonging to this view
    for (clustering::iterator c_itr = _c[m].begin();
            c_itr != _c[m].end();
            c_itr++) {
        CRP& cluster = c_itr->second;
        log_lik += gammaln(_eta_sum) - gammaln(_eta_sum + cluster.nwsum) +
            gammaln(_eta[w]+cluster.nw[w]) - gammaln(_eta[w]);
    }

    // Prior over the featurecluster assignments
    log_lik += log(_b[m].ndsum + _xi[m]) - log(_lV+_xi_sum);
    // log_lik += log(_b[m].ndsum) - log(_lV - 1 + FLAGS_cc_xi);

    return log_lik;
}

double CrossCatMM::cross_cat_reassign_features(unsigned old_m, unsigned new_m, unsigned w) {
    if (old_m != new_m) {
        // Update counts
        _m[w] = new_m;
        _b[old_m].ndsum -= 1;  // NOTE: comes after cluster likelihood b/c MH
        _b[new_m].ndsum += 1;

        // reassign the words temporarily; this has to loop over documents because
        // we need to change what clusters the features get assigned to maybe
        for (int d = 0; d < _DD.size(); d++) {
            unsigned zdm_old = _z[d][old_m];
            unsigned zdm_new = _z[d][new_m];

            collapsed_document::iterator d_itr = _DD[d].find(w);
            if (d_itr != _DD[d].end()) {
                unsigned count = d_itr->second;
                _c[old_m][zdm_old].nw[w] -= count;
                _c[old_m][zdm_old].nwsum -= count;
                
                CHECK_GE(_c[old_m][zdm_old].nw[w], 0);
                CHECK_GE(_c[old_m][zdm_old].nwsum, 0);

                _c[new_m][zdm_new].nw[w] += count;
                _c[new_m][zdm_new].nwsum += count;
            }
        }

        // Bookkeeping for the per-view feature marginals.
        CHECK_EQ(_b[new_m].nw[w], 0);
        _b[new_m].nw[w] = _b[old_m].nw[w];
        _b[old_m].nw[w] = 0;
    }
}

void CrossCatMM::resample_posterior_m_for(unsigned w) {
    if (true) {
        // Choose a new random cluster
        unsigned new_m = sample_integer(_b.size());
        unsigned old_m = _m[w];

        VLOG(1) << "proposing to move [" << _word_id_to_name[w] << "] from " << old_m << " to " << new_m;

        // Do a MH step
        if (new_m != _m[w]) {
            // XXX TODO: only need to compute ll over things that change
            double current_likelihood = cross_cat_clustering_log_likelihood(w, _m[w]);

            VLOG(1) << "current likelihood: " << current_likelihood;
                
            cross_cat_reassign_features(old_m, new_m, w);

            double new_likelihood = cross_cat_clustering_log_likelihood(w, _m[w]);

            VLOG(1) << "new likelihood: " << new_likelihood;

            if (log(sample_uniform()) < new_likelihood - current_likelihood) {
                VLOG(1) << "MOVING [" << _word_id_to_name[w] << "] from " << old_m << " to " << new_m;
            } else {
                cross_cat_reassign_features(new_m, old_m, w);
                _m_failed += 1;
            }

            _m_proposed += 1;
        }
    } else {
        // XXX: ratios of Dirichlet Processes won't really work; need to think
        // through the math a little bit more. The problem is that the original
        // clustering has a different number of clusters than the new one, and
        // aynway it doesn't make sense to loop over cluster pairs. Really we
        // need to integrate out the clustersings (marginalize over all possible
        // clusterings in both cases). But this is unfortunately prohibitively
        // expensive for DPMM.
        /*
        // Gibbs sampler for XCAT?
        unsigned old_m = _m[w];

        // Loop over all the documents and generate a hash of
        // cluster->w_count for the old view (i.e. count deltas if w was
        // removed from this view.
        hash_map<unsigned, double> cluster_to_w_count;
        for (int d = 0; d < _DD.size(); d++) {
            collapsed_document::iterator d_itr = _DD[d].find(w);
            if (d_itr != _DD[d].end()) {
                cluster_to_w_count[_z[d][new_m]] += d_itr->second;
            }
        }


        vector<pair<unsigned,double> > lp_m;
        for (int new_m = 0; new_m < _b.size(); new_m++) {
            double log_lik = 0;

            // cross_cat_reassign_features(old_m, new_m, w);

            // Loop over all the documents and generate a hash of
            // cluster->w_count for the new_m view (i.e. count deltas if w was
            // assigned to this view.
            hash_map<unsigned, double> cluster_to_w_count;
            for (int d = 0; d < _DD.size(); d++) {
                collapsed_document::iterator d_itr = _DD[d].find(w);
                if (d_itr != _DD[d].end()) {
                    cluster_to_w_count[_z[d][new_m]] += d_itr->second;
                }
            }

            // TODO: XXX: the below is probably wrong; should be a ratio of C()s 
            
            // Likelihood of this particular feature w belonging to this view
            for (clustering::iterator c_itr = _c[new_m].begin();
                    c_itr != _c[new_m].end();
                    c_itr++) {
                unsigned c_id = c_itr->first;
                CRP& cluster = c_itr->second;

                if (new_m == old_m) {
                    log_lik += gammaln(_eta_sum) - gammaln(_eta_sum + cluster.nwsum) +
                        gammaln(_eta[w]+cluster.nw[w]) - gammaln(_eta[w]);
                } else {
                    CHECK_EQ(cluster.nw[w], 0);
                    // x is the number of w for all docs in this cluster 
                    unsigned x = cluster_t_w_count[c_id];

                    log_lik += gammaln(_eta_sum) - gammaln(_eta_sum + cluster.nwsum) +
                        gammaln(_eta[w]+cluster.nw[w]) - gammaln(_eta[w]);
                }

            }

            // Prior over the featurecluster assignments
            // if (FLAGS_mm_prior == kDirichletMixture) {
            if (new_m == old_m) {
                log_lik += log(_b[new_m].ndsum+_xi[new_m]) - log(_lV+_xi_sum);
            } else {
                log_lik += log(_b[new_m].ndsum+1+_xi[new_m]) - log(_lV+_xi_sum);
            }
            // } else if (FLAGS_mm_prior == kDirichletProcess) {
            //     log_lik += log(_b[new_m].ndsum) - log(_lV - 1 + FLAGS_cc_xi);
            // }

            // cross_cat_reassign_features(new_m, old_m, w);

            lp_m.push_back(pair<unsigned,double>(new_m, log_lik));
        }

        // Update the assignment
        _m[w] = sample_unnormalized_log_multinomial(&lp_m);
        if (_m[w] != old_m) {
            VLOG(1) << "MOVING [" << _word_id_to_name[w] << "] from " << old_m << " to " << _m[w];
            cross_cat_reassign_features(old_m, _m[w], w);
        } else {
            _m_failed += 1;
        }
        _m_proposed += 1;*/
    }
}

double CrossCatMM::compute_log_likelihood() {
    // Compute the log likelihood for the tree
    double log_lik = 0;

    // Compute the log likelihood of the level assignments (correctly?)
    for (multiple_clustering::iterator c_itr = _c.begin();
            c_itr != _c.end();
            c_itr++) { 
        unsigned m = c_itr->first;
        log_lik += compute_log_likelihood_for(m, c_itr->second);

        // Prior over the featurecluster assignments
        // if (FLAGS_mm_prior == kDirichletMixture) {
        log_lik += log(_b[m].ndsum+_xi[m]) - log(_lV+_xi_sum);
        // } else if (FLAGS_mm_prior == kDirichletProcess) {
        //     log_lik += log(_b[m].ndsum) - log(_lV - 1 + FLAGS_cc_xi);
        // }
    }

    return log_lik;
}

double CrossCatMM::compute_log_likelihood_for(unsigned m, clustering& cm) {
    double log_lik = 0;
    for (clustering::iterator c_itr = cm.begin();
            c_itr != cm.end();
            c_itr++) {
        unsigned cluster_id = c_itr->first;
        CRP& cluster = c_itr->second;

        // Likelihood of all the words | the clustering cm in view m
        log_lik += gammaln(_eta_sum) - gammaln(_eta_sum + cluster.nwsum);
        for (WordToCountMap::iterator w_itr = cluster.nw.begin();
                w_itr != cluster.nw.end();
                w_itr++) {
            unsigned w = w_itr->first;
            unsigned count = w_itr->second;
            if (_m[w] == m) {
                log_lik += gammaln(_eta[w]+count) - gammaln(_eta[w]);
            }
        }

        // Likelihood of document in the clustering / view
        log_lik += log(cluster.ndsum) - log(_lD - 1 + FLAGS_mm_alpha);
    }
    return log_lik;
}


string CrossCatMM::current_state() {
    _output_filename = FLAGS_mm_datafile;
    _output_filename += StringPrintf("-alpha%f-eta%f-xi%f",
            FLAGS_mm_alpha,
            _eta_sum / (double)_eta.size(),
            _xi_sum / (double)_xi.size());

    // Compute a vector of the clsuter sizes for printing.
    vector<string> cluster_sizes;
    for (multiple_clustering::iterator c_itr = _c.begin();
        c_itr != _c.end();
        c_itr++) {
        unsigned m = c_itr->first;
        clustering& cm = c_itr->second;
        cluster_sizes.push_back(StringPrintf("[view: %d | clusters: %d | features: %d]" , m, cm.size(), _b[m].ndsum));
    }

    return StringPrintf(
            "ll = %f (%f at %d) alpha = %f eta = %f xi = %f K = %s M = %d",
            _ll, _best_ll, _best_iter,
            FLAGS_mm_alpha,
            _eta_sum / (double)_eta.size(),
            _xi_sum / (double)_xi.size(),
            JoinStrings(cluster_sizes, " ").c_str(),
            _b.size());
}

void CrossCatMM::resample_posterior() {
    CHECK_GT(_lV, 0);
    CHECK_GT(_lD, 0);
    CHECK_GT(_c.size(), 0);


    // Resample the feature clustering using MH
    if (FLAGS_M > 1) {

        if (_iter % 10 == 0) {
            LOG(INFO) << "reallocating 10p features...";
            resample_posterior_m(0.1);
        } else {
            resample_posterior_m(FLAGS_cc_feature_move_rate);
        }
    }

    // Resample the cluster indicators
    _c_proposed = _c_failed = 0;
    for (int d = 0; d < _DD.size(); d++) {
        // LOG(INFO) <<  "  resampling document " <<  d;
        // For each clustering (view), resample this document's cluster indicator
        for (multiple_clustering::iterator c_itr = _c.begin();
                c_itr != _c.end();
                c_itr++) { 
            resample_posterior_z_for(d, c_itr->first, true);
        }
    }
    LOG(INFO) << _c_proposed-_c_failed << " / " << _c_proposed << " " 
        << StringPrintf("(%.3f%%)", 100 - _c_failed / (double)_c_proposed*100) << " cluster moves.";

    // Write the current cluster sizes to the console
    for (multiple_clustering::iterator c_itr = _c.begin();
            c_itr != _c.end();
            c_itr++) { 
        unsigned m = c_itr->first;
        clustering& cm = _c[m];

        // Summarize what features are assigned to this view
        LOG(INFO) << "M[" << m << "] (f " << _b[m].ndsum << ") " 
                  << " " << show_chopped_sorted_nw(_b[m].nw);

        // Show the contents of the clusters
        unsigned test_sum = 0;
        for (clustering::iterator itr = cm.begin();
                itr != cm.end();
                itr++) {
            unsigned l = itr->first;

            // Compute the number of documents in this cluster with actual
            // features in the game
            unsigned non_zero_docs = 0;
            for (int d = 0; d < _DD.size(); d++) {
                for (collapsed_document::iterator d_itr = _DD[d].begin();
                        d_itr != _DD[d].end();
                        d_itr++) {
                    unsigned w = d_itr->first;

                    if (_m[w] == m && _z[d][m] == l) {
                        non_zero_docs += 1;
                        break;
                    }
                }
            }

            // LOG(INFO) << "  C[" << l << "] (d " << itr->second.ndsum << ") " 
            //           << " " << show_chopped_sorted_nw(itr->second.nw);
            LOG(INFO) << "  C[" << l << "] (d " << non_zero_docs << ") " 
                      << " " << show_chopped_sorted_nw(itr->second.nw);

            test_sum += cm[l].ndsum;
        }
        CHECK_EQ(test_sum,_lD);  // make sure we haven't lost any docs
    }

    // Write out how the features are mapped to views
    for (WordCode::iterator itr = _word_id_to_name.begin();
            itr != _word_id_to_name.end();
            itr++) {
        unsigned w = itr->first;
        string word = itr->second;

        VLOG(1) << word << "\t" << _m[w];
    }

}

// Write out all the data in an intermediate format
void CrossCatMM::write_data(string prefix) {
    // string filename = StringPrintf("%s-%d-%s.hlda.bz2", get_base_name(_output_filename).c_str(), FLAGS_random_seed,
    //         prefix.c_str());
    string filename = StringPrintf("%s-%d-%s.hlda", get_base_name(_output_filename).c_str(),
            FLAGS_random_seed, prefix.c_str());
    VLOG(1) << "writing data to [" << filename << "]";

    // XXX: Order in which f_raw and filtering ostream are declared matters.
    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    // f = get_bz2_ostream(filename);

    f << current_state() << endl;

    // Write out how each document gets clustered in each view.
    for (multiple_clustering::iterator c_itr = _c.begin();
            c_itr != _c.end();
            c_itr++) { 
        unsigned m = c_itr->first;
        clustering& cm = c_itr->second;
        for (int d = 0; d < _DD.size(); d++) {
            f << d << "\t" << m << "\t" << _z[d][m] << endl;
        }
    }

    // Write out how the features are mapped to views
    for (WordCode::iterator itr = _word_id_to_name.begin();
            itr != _word_id_to_name.end();
            itr++) {
        unsigned w = itr->first;
        string word = itr->second;

        f << word << "\t" << _m[w] << endl;
    }
    //boost::iostreams::flush(f);
    VLOG(1) << "done";
}

