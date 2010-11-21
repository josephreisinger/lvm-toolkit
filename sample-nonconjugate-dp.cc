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

// the number of clusters
DEFINE_int32(K,
             50,
             "Number of clusters.");

// the number of feature clusters
DEFINE_int32(M,
             1,
             "Number of feature clusters / data views.");

// Prior over cluster sizes
DEFINE_string(mm_prior,
              "dirichlet",
              "Prior over clusters. Can be dirichlet, dirichlet-process, or uniform.");

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
DEFINE_int32(cc_feature_moves,
             -1,
             "Number of MH moves (features) to make per Gibbs sweep (-1 means 10% V).");

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

void CrossCatMM::initialize() {
    LOG(INFO) << "initialize";

    CHECK(FLAGS_mm_prior == kDirichletMixture ||
          FLAGS_mm_prior == kDirichletProcess ||
          FLAGS_mm_prior == kUniformMixture);

    CHECK(!FLAGS_cc_include_noise_view || FLAGS_K > 1);
    
    // _DD.set_empty_key(kEmptyUnsignedKey);

    _b.set_empty_key(kEmptyUnsignedKey);
    _m.set_empty_key(kEmptyUnsignedKey);

    // Set up the asymmetric dirichlet priors for each clustering and for the
    // cross-cat
    LOG(INFO) << "got K=" << FLAGS_K << " clusters.";
    for (int l = 0; l < FLAGS_K; l++) {
        _alpha.push_back(FLAGS_mm_alpha);
    }
    _alpha_sum = FLAGS_K*FLAGS_mm_alpha;

    for (int l = 0; l < FLAGS_M; l++) {
        _xi.push_back(FLAGS_cc_xi);
    }
    _xi_sum = FLAGS_M*FLAGS_cc_xi;

    _eta.clear();
    for (int l = 0; l < _lV; l++) {
        _eta.push_back(FLAGS_eta);
    }
    _eta_sum = _lV*FLAGS_eta;


    _current_component.resize(FLAGS_M);

    // Allocate the initial clusters
    for (int m = 0; m < FLAGS_M; m++) {
        clustering t;
        t.set_empty_key(kEmptyUnsignedKey);
        t.set_deleted_key(kDeletedUnsignedKey);
        _c.insert(pair<unsigned,clustering>(m, t));

        // If we're using the noise view, then only push back a single cluster
        // on the first view
        if (FLAGS_cc_include_noise_view && m == 0) {
            _c[m].insert(pair<unsigned,CRP>(0, CRP()));
            _current_component[m] = 1;
        } else {
            for (int l = 0; l < FLAGS_K; l++) {
                _c[m].insert(pair<unsigned,CRP>(l, CRP()));
            }
            _current_component[m] = FLAGS_K;
        }

        _b.insert(pair<unsigned,CRP>(m, CRP()));
    }

    // Allocate the initial feature clusters / mappings
    for (int w = 0; w < _lV; w++) {
        _m[w] = sample_integer(FLAGS_M);
        _b[_m[w]].nd[0] += 1;

        VLOG(1) << _word_id_to_name[w] << " assigned to view " << _m[w];
    }

    // Add the documents into the clustering
    for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;  // = document number

        cluster_map t;
        t.set_empty_key(kEmptyUnsignedKey);
        _z.insert(pair<unsigned,cluster_map>(d, t));
        // _z[d].set_empty_key(kEmptyUnsignedKey);

        // set a random topic assignment for this guy
        for (int m = 0; m < FLAGS_M; m++) {
            _z[d][m] = sample_integer(_c[m].size());
            _c[m][_z[d][m]].nd[0]    += 1;  // # of words in doc d with topic z
        }

        // Allocate this doc in the collapsed vector
        collapsed_document temp;
        // temp.set_empty_key(kEmptyUnsignedKey);
        _DD.insert(pair<unsigned,collapsed_document>(d, temp));

        // Initial level assignments
        for (int n = 0; n < d_itr->second.size(); n++) {
            unsigned w = d_itr->second[n];

            unsigned zdm = _z[d][_m[w]];

            // test the initialization of maps
            CHECK(_c[_m[w]][zdm].nw.find(w) != _c[_m[w]][zdm].nw.end()
                    || _c[_m[w]][zdm].nw[w] == 0);

            _c[_m[w]][zdm].nw[w] += 1;  // # of words in topic z equal to w
            _c[_m[w]][zdm].nwsum += 1;  // # of words in topic z

            _DD[d][w] += 1;
        }
    }
    // Cull the DP assignments if there are any unused initial clusters.
    if (FLAGS_mm_prior == kDirichletProcess) {
        for (int m = 0; m < FLAGS_M; m++) {
            for (clustering::iterator itr = _c[m].begin();
                    itr != _c[m].end();
                    itr++) {
                if (itr->second.nd[0] == 0) {  // empty component
                    _c[m].erase(itr);
                }
            }
        }
    }
    
    if (FLAGS_cc_feature_moves == -1) {
        FLAGS_cc_feature_moves = max((unsigned)1, _lV / 100);
        LOG(INFO) << "performing " << FLAGS_cc_feature_moves << " feature moves per iteration.";
    }

    _ll = compute_log_likelihood();
    
}

// Performs a single document's level assignment resample step
void CrossCatMM::resample_posterior_z_for(unsigned d) {
    for (multiple_clustering::iterator c_itr = _c.begin();
            c_itr != _c.end();
            c_itr++) { 
        unsigned m = c_itr->first;
        clustering& cm = c_itr->second;

        unsigned old_zdm = _z[d][m];

        // Remove this document from all the clusterings
        unsigned total_count = 0;
        for (collapsed_document::iterator d_itr = _DD[d].begin();
                d_itr != _DD[d].end();
                d_itr++) {
            unsigned w = d_itr->first;
            unsigned count = d_itr->second;
            if (_m[w] == m) {
                // Remove this document and word from the counts
                cm[old_zdm].nw[w] -= count;  // # of words in topic z equal to w

                CHECK_GE(cm[old_zdm].nw[w], 0);

                total_count += count;
            }
        }
        CHECK_GT(cm[old_zdm].nd[0], 0);

        cm[old_zdm].nwsum -= total_count;  // # of words in topic z
        cm[old_zdm].nd[0] -= 1;  // # of docs in topic

        CHECK_GE(cm[old_zdm].nwsum, 0);
        CHECK_LE(cm[old_zdm].nd[0], _total_doc_count);

        vector<pair<unsigned,double> > lp_z_d;

        for (clustering::iterator itr = cm.begin();
                itr != cm.end();
                itr++) {
            unsigned l = itr->first;

            double sum = 0;
            
            // First add in the prior over the clusters
            if (FLAGS_mm_prior == kDirichletMixture) {
                sum += log(_alpha[l] + cm[l].nd[0]) - log(_alpha_sum + _total_doc_count);
            } else if (FLAGS_mm_prior == kDirichletProcess) {
                sum += log(cm[l].nd[0]) - log(_total_doc_count - 1 + FLAGS_mm_alpha);
            }

            // Now account for the likelihood of the data (marginal posterior of
            // DP-Mult)
            for (collapsed_document::iterator d_itr = _DD[d].begin();
                    d_itr != _DD[d].end();
                    d_itr++) {
                unsigned w = d_itr->first;
                unsigned count = d_itr->second;

                if (_m[w] == m) {
                    // check that ["doesnt exist"]->0
                    // DCHECK(cm[l].nw.find(w) != cm[l].nw.end() || cm[l].nw[w] == 0);

                    sum += count*(log(_eta[w] + cm[l].nw[w]) - log(_eta_sum + cm[l].nwsum));
                }
            }
            lp_z_d.push_back(pair<unsigned,double>(l, sum));
        }

        // Add an additional new component if DP
        if (FLAGS_mm_prior == kDirichletProcess && (m != 0 || !FLAGS_cc_include_noise_view)) {
            double sum = log(FLAGS_mm_alpha) - log(_total_doc_count - 1 + FLAGS_mm_alpha);
            for (collapsed_document::iterator d_itr = _DD[d].begin();
                    d_itr != _DD[d].end();
                    d_itr++) {
                unsigned w = d_itr->first;
                unsigned count = d_itr->second;
                if (_m[w] == m) {
                    sum += count*(log(_eta[w]) - log(_eta_sum));
                }
            }

            lp_z_d.push_back(pair<unsigned,double>(_current_component[m], sum));
        }

        // Update the assignment
        _z[d][m] = sample_unnormalized_log_multinomial(&lp_z_d);
        VLOG(1) << "resampling posterior z for " << d << "," << m << ": " << old_zdm << "->" << _z[d][m];

        unsigned new_zdm = _z[d][m];

        // Update the counts
        for (collapsed_document::iterator d_itr = _DD[d].begin();
                d_itr != _DD[d].end();
                d_itr++) {
            unsigned w = d_itr->first;
            unsigned count = d_itr->second;

            if (_m[w] == m) {
                // Check to see that the default dictionary insertion works like we
                // expect
                // DCHECK(cm[new_zdm].nw.find(w) != cm[new_zdm].nw.end()
                //       || cm[new_zdm].nw[w] == 0);

                cm[new_zdm].nw[w] += count;  // number of words in topic z equal to w
            }
        }
        cm[new_zdm].nwsum += total_count;  // number of words in topic z
        cm[new_zdm].nd[0] += 1;  // number of words in doc d with topic z

        CHECK_LE(cm[new_zdm].nd[0], _total_doc_count);

        // Clean up for the DPCrossCatMM
        if (FLAGS_mm_prior == kDirichletProcess) {
            if (cm[old_zdm].nd[0] == 0) {  // empty component
                _c.erase(old_zdm);
            }
            // Make room for a new component if we selected the new one
            if (new_zdm == _current_component[m]) {
                _current_component[m] += 1;
            }
        }
    }
}

void CrossCatMM::resample_posterior_m() {
    VLOG(1) << "resampling m using MH";

    // Choose a random feature
    unsigned tw = sample_integer(_lV);

    // Do the resampling
    resample_posterior_m_for(tw);
}

// Likelihood of a particular cross-cat clustering
double CrossCatMM::cross_cat_clustering_log_likelihood(unsigned w, unsigned m) {
    double log_lik = 0;
    for (int d = 0; d < _DD.size(); d++) {
        unsigned zdm = _z[d][m];

        collapsed_document::iterator d_itr = _DD[d].find(w);
        if (d_itr != _DD[d].end()) {
            unsigned count = d_itr->second;
            log_lik += count*(log(_c[m][zdm].nw[w]+_eta[w]) - log(_c[m][zdm].nwsum+_eta_sum));
        }
    }

    if (FLAGS_mm_prior == kDirichletMixture) {
        log_lik += log(_b[m].nd[0]+_xi[m]) - log(_lV+_xi_sum);
    } else if (FLAGS_mm_prior == kDirichletProcess) {
        log_lik += log(_b[m].nd[0]) - log(_lV - 1 + FLAGS_cc_xi);
    }
    return log_lik;
}

double CrossCatMM::cross_cat_reassign_features(unsigned old_m, unsigned new_m, unsigned w) {
    // Update counts
    _m[w] = new_m;
    _b[old_m].nd[0] -= 1;  // NOTE: comes after cluster likelihood b/c MH
    _b[new_m].nd[0] += 1;

    // XXX: reassign the words temporarily
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
}

void CrossCatMM::resample_posterior_m_for(unsigned w) {
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
        }
    }
}

double CrossCatMM::compute_log_likelihood() {
    // Compute the log likelihood for the tree
    double log_lik = 0;

    // Compute the log likelihood of the level assignments (correctly?)
    for (multiple_clustering::iterator c_itr = _c.begin();
            c_itr != _c.end();
            c_itr++) { 
        log_lik += compute_log_likelihood_for(c_itr->first, c_itr->second);

        // TODO XXX Likelihood of clustering
    }

    return log_lik;
}

double CrossCatMM::compute_log_likelihood_for(unsigned m, clustering& cm) {
    double log_lik = 0;
    for (int d = 0; d < _DD.size(); d++) {
        unsigned zdm = _z[d][m];
        // Remove this document from all the clusterings
        unsigned total_count = 0;
        for (collapsed_document::iterator d_itr = _DD[d].begin();
                d_itr != _DD[d].end();
                d_itr++) {
            unsigned w = d_itr->first;
            unsigned count = d_itr->second;

            if (_m[w] == m) {
                log_lik += count*(log(cm[zdm].nw[w]+_eta[w]) - log(cm[zdm].nwsum+_eta_sum));
            }
        }
        // Likelihood of document in the clustering / view
        if (FLAGS_mm_prior == kDirichletMixture) {
            log_lik += log(cm[zdm].nd[0]+_alpha[zdm]) - log(_total_doc_count+_alpha_sum);
        } else if (FLAGS_mm_prior == kDirichletProcess) {
            log_lik += log(cm[zdm].nd[0]) - log(_total_doc_count - 1 + FLAGS_mm_alpha);
        }
    }
    return log_lik;
}


string CrossCatMM::current_state() {
    _output_filename = FLAGS_mm_datafile;
    _output_filename += StringPrintf("-alpha%f-eta%f-xi%f",
            _alpha_sum / (double)_alpha.size(),
            _eta_sum / (double)_eta.size(),
            _xi_sum / (double)_xi.size());

    // Compute a vector of the clsuter sizes for printing.
    vector<string> cluster_sizes;
    for (multiple_clustering::iterator c_itr = _c.begin();
        c_itr != _c.end();
        c_itr++) {
        unsigned m = c_itr->first;
        clustering& cm = c_itr->second;
        cluster_sizes.push_back(StringPrintf("[view: %d clusters: %d features: %d]" , m, cm.size(), _b[m].nd[0]));
    }

    return StringPrintf(
            "ll = %f (%f at %d) alpha = %f eta = %f xi = %f K = %s M = %d",
            _ll, _best_ll, _best_iter,
            _alpha_sum / (double)_alpha.size(),
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
        for (int m = 0; m < FLAGS_cc_feature_moves; m++) {
            resample_posterior_m();
        }
    }

    if (FLAGS_K > 1) {
        for (int d = 0; d < _DD.size(); d++) {
            // LOG(INFO) <<  "  resampling document " <<  d;
            resample_posterior_z_for(d);
        }
    }

    // Write the current cluster sizes to the console
    for (multiple_clustering::iterator c_itr = _c.begin();
            c_itr != _c.end();
            c_itr++) { 
        string s;
        unsigned m = c_itr->first;
        clustering& cm = c_itr->second;
        unsigned test_sum = 0;
        for (clustering::iterator itr = cm.begin();
                itr != cm.end();
                itr++) {
            unsigned l = itr->first;
            s += StringPrintf("%d:%d ", l, cm[l].nd[0]);
            test_sum += cm[l].nd[0];
        }
        LOG(INFO) << m << " :: " << s;
        CHECK_EQ(test_sum,_total_doc_count);  // make sure we haven't lost any docs
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
    string filename = StringPrintf("%s-%s-%d-%s.hlda", get_base_name(_output_filename).c_str(),
            FLAGS_mm_prior.c_str(), FLAGS_random_seed,
            prefix.c_str());
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

