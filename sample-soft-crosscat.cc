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

#include <sys/stat.h>

#include <algorithm>


#include "gibbs-base.h"
#include "sample-soft-crosscat.h"

// Two main implementations:
//   (1) normal: treat the topic model part for a document as the set of
//       clusters picked out by the document (one per view)
//   (2) marginal: treat each "topic" as the marginal over all clusters inside
//       it; this model seems to make more sense to me
DEFINE_string(implementation,
             "normal",
             "_normal_ model or cluster _marginal_ model.");

// the number of feature clusters
DEFINE_int32(M,
             1,
             "Number of feature clusters / data views.");

// the maximum number of clusters
DEFINE_int32(KMAX,
             -1,
             "Maximum number of clusters.");

// Smoother on clustering
DEFINE_double(mm_alpha,
              1.0,
              "Smoother on the cluster assignments.");

// Smoother on cross cat clustering
DEFINE_double(cc_xi,
              1.0,
              "Smoother on the cross-cat cluster assignments.");

// File holding the data
DEFINE_string(mm_datafile,
              "",
              "Docify holding the data to be clustered.");

// If toggled, the first view will be constrained to a single cluster
DEFINE_bool(cc_include_noise_view,
            false,
            "Should the first view be confined to a single cluster.");

// If toggled, will resume from the best model written so far
DEFINE_bool(cc_resume_from_best,
            false,
            "Should we try to resume from a file?");

// If defined, will load the seed file 
DEFINE_string(cc_fixed_topic_seed,
              "",
              "File to seed from, if any.");

const string kNormalModel = "normal";
const string kMarginalModel = "marginal";

void SoftCrossCatMM::clean_initialization() {
    _iter = _best_iter = 0;
    // Keep track of the number of clusters in each view
    _current_component.clear();
    _current_component.resize(FLAGS_M);

    // Allocate the initial clusters
    _cluster.clear();
    _cluster_marginal.clear();
    for (int m = 0; m < FLAGS_M; m++) {
        clustering t;
        t.set_empty_key(kEmptyUnsignedKey);
        t.set_deleted_key(kDeletedUnsignedKey);
        _cluster.insert(pair<unsigned,clustering>(m, t));

        // Push back a single cluster; we'll allocate more potentially later
        _cluster[m].insert(pair<unsigned,CRP>(0, CRP()));
        _current_component[m] = 1;

    }

    // Initialize _c and _z
    _lD = 0;  // reset this to make the resample_posterior_z stuff below work correctly
    for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;  // = document number
        _lD += 1;

        cluster_map tc;
        tc.set_empty_key(kEmptyUnsignedKey);
        _c.insert(pair<unsigned,cluster_map>(d, tc));

        cluster_map tz;
        tz.set_empty_key(kEmptyUnsignedKey);
        _z.insert(pair<unsigned,cluster_map>(d, tz));
    }
}

void SoftCrossCatMM::batch_allocation() {
    CHECK(FLAGS_implementation == "normal" || FLAGS_implementation == "marginal");

    is_cluster_marginal = (FLAGS_implementation == "marginal");
    is_fixed_topics = (FLAGS_cc_fixed_topic_seed != "");

    // Set up the asymmetric dirichlet priors for each clustering and for the
    // cross-cat
    // _c.set_empty_key(kEmptyUnsignedKey);
    // _z.set_empty_key(kEmptyUnsignedKey);

    _cluster_marginal.set_empty_key(kEmptyUnsignedKey);
    _cluster_marginal.set_deleted_key(kDeletedUnsignedKey);

    if (!FLAGS_cc_resume_from_best || !restore_data_from_prefix("best")) {
        LOG(INFO) << "clean initialize";

        // Clean out the data structures just in-case we added anything
        clean_initialization();

        // Load seed file if necessary
        if (is_fixed_topics) {
            restore_data_from_file(FLAGS_cc_fixed_topic_seed);
        }

        // Add the documents into the clustering
        for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
            unsigned d = d_itr->first;  // = document number

            // set a random topic/crosscut view assignment for this document
            for (int m = 0; m < FLAGS_M; m++) {
                _c[d][m] = 0; //sample_integer(_cluster[m].size());
                _cluster[m][_c[d][m]].ndsum += 1; // Can't use ADD b/c we need to maintain ndsum over all the views
                _cluster_marginal[m].ndsum += 1; // Can't use ADD b/c we need to maintain ndsum over all the views
            }

            // Initial level assignments
            for (int n = 0; n < _D[d].size(); n++) {
                unsigned w = _D[d][n];

                _z[d][n] = sample_integer(FLAGS_M);

                // _m maps a document-term to a view
                unsigned zdn = _z[d][n];
                // _z maps a document-view to a cluster
                unsigned cdm = _c[d][zdn];

                // test the initialization of maps
                CHECK(_cluster[zdn][cdm].nw.find(w) != _cluster[zdn][cdm].nw.end() || _cluster[zdn][cdm].nw[w] == 0);

                // Cant use ADD b/c we need to keep the ndsum consistent across the
                // views
                _cluster[zdn][cdm].add_no_ndsum(w, d);
                _cluster_marginal[zdn].add_no_ndsum(w, d);
            }
            
            if (d > 0) {
                resample_posterior_c_for(d);
                resample_posterior_z_for(d);
            }

            if (d % 1000 == 0 && d > 0) {
                string cluster_sizes = "";
                for (int m = 0; m < _cluster.size(); m++) {
                    cluster_sizes += StringPrintf("%d ", _cluster[m].size());
                }

                LOG(INFO) << "Sorted " << d << " documents into " << FLAGS_M << " views, sized: " << cluster_sizes;
            }
        }
    }
    
    _ll = compute_log_likelihood();
    
}

// Performs a single document's level assignment resample step
void SoftCrossCatMM::resample_posterior_c_for(unsigned d) {
    for (int m = 0; m < FLAGS_M; m++) {
        unsigned old_cdm = _c[d][m];

        unsigned total_removed_count = 0;
        google::dense_hash_map<unsigned, unsigned> removed_w;
        removed_w.set_empty_key(kEmptyUnsignedKey);
        
        // Remove this document from this clustering
        for (int n = 0; n < _D[d].size(); n++) {
            if (_z[d][n] == m) {
                unsigned w = _D[d][n];
                total_removed_count += 1;
                removed_w[w] += 1;
                // Remove this document and word from the counts
                _cluster[m][old_cdm].nw[w] -= 1;  // # of words in cluster c view m
                _cluster[m][old_cdm].nwsum -= 1;  // # of words in cluster c
                _cluster_marginal[m].nw[w] -= 1;
                _cluster_marginal[m].nwsum -= 1;
                CHECK_GE(_cluster[m][old_cdm].nw[w], 0);
                CHECK_GE(_cluster_marginal[m].nw[w], 0);
            }
        }
        CHECK_GT(_cluster[m][old_cdm].ndsum, 0);

        _cluster[m][old_cdm].ndsum -= 1;  // # of docs in clsuter

        CHECK_GE(_cluster[m][old_cdm].nwsum, 0);
        CHECK_LT(_cluster[m][old_cdm].ndsum, _lD);

        // Compute the log likelihood of each cluster assignment given all the other
        // document-cluster assignments
        vector<pair<unsigned,double> > lp_z_d;
        for (clustering::iterator itr = _cluster[m].begin();
                itr != _cluster[m].end();
                itr++) {
            unsigned l = itr->first;

            double sum = 0;
            
            // First add in the prior over the clusters
            sum += log(_cluster[m][l].ndsum) - log(_lD - 1 + FLAGS_mm_alpha);

            // Add in the normalizer for the multinomial-dirichlet likelihood
            sum += gammaln(_eta_sum + _cluster[m][l].nwsum) - gammaln(_eta_sum + _cluster[m][l].nwsum + total_removed_count);

            // Now account for the likelihood of the data (marginal posterior of
            // DP-Mult); only need to loop over what was actually removed since
            // other stuff (removed_w = 0) ends up canceling the two gammalns
            for (google::dense_hash_map<unsigned,unsigned>::iterator itr = removed_w.begin();
                    itr != removed_w.end();
                    itr++) {
                unsigned w = itr->first;
                unsigned count = itr->second;
                sum += gammaln(_eta[w] + count + _cluster[m][l].nw[w]) - gammaln(_eta[w] + _cluster[m][l].nw[w]);
            }
            lp_z_d.push_back(pair<unsigned,double>(l, sum));
        }


        // Add an additional new component if not in the noise view, we haven't
        // hit KMAX, and this isn't a singleton cluster already
        if (_cluster[m][old_cdm].ndsum > 0) {
            if (_cluster[m].size() < FLAGS_KMAX || FLAGS_KMAX==-1) {
                if (m != 0 || !FLAGS_cc_include_noise_view) {
                    double sum = 0;
                    sum += log(FLAGS_mm_alpha) - log(_lD - 1 + FLAGS_mm_alpha);

                    // Add in the normalizer for the multinomial-dirichlet likelihood
                    sum += gammaln(_eta_sum) - gammaln(_eta_sum + total_removed_count);
                    for (google::dense_hash_map<unsigned,unsigned>::iterator itr = removed_w.begin();
                            itr != removed_w.end();
                            itr++) {
                        unsigned w = itr->first;
                        unsigned count = itr->second;
                        sum += gammaln(_eta[w] + count) - gammaln(_eta[w]);
                    }
                    lp_z_d.push_back(pair<unsigned,double>(_current_component[m], sum));
                }
            }
        }

        // Update the assignment
        _c[d][m] = sample_unnormalized_log_multinomial(&lp_z_d);
        VLOG(1) << "resampling posterior c for " << d << "," << m << ": " << old_cdm << "->" << _c[d][m];

        unsigned new_cdm = _c[d][m];

        if (new_cdm == old_cdm) {
            _c_failed += 1;
        }
        _c_proposed += 1;

        // Update the counts
        for (google::dense_hash_map<unsigned,unsigned>::iterator itr = removed_w.begin();
                itr != removed_w.end();
                itr++) {
            unsigned w = itr->first;
            unsigned count = itr->second;
            _cluster[m][new_cdm].nw[w] += count;  // number of words in topic z equal to w
            _cluster_marginal[m].nw[w] += count;  // number of words in topic z equal to w
        }
        _cluster[m][new_cdm].nwsum += total_removed_count;  // number of words in topic z
        _cluster[m][new_cdm].ndsum += 1;  // number of words in doc d with topic z

        _cluster_marginal[m].nwsum += total_removed_count;  // number of words in topic z
        _cluster_marginal[m].ndsum += 1;  // number of words in doc d with topic z

        CHECK_LE(_cluster[m][new_cdm].ndsum, _lD);

        // Clean up for the DPSoftCrossCatMM
        if (_cluster[m][old_cdm].ndsum == 0) {  // empty component
            // LOG(INFO) << "removing cluster " << old_zdm << " from view "  << m << " because we chose cluster " << new_zdm;
            // _c.erase(old_zdm);
            _cluster[m].erase(old_cdm);
        }
        // Make room for a new component if we selected the new one
        if (new_cdm == _current_component[m]) {
            _current_component[m] += 1;
        }
    }
}

void SoftCrossCatMM::resample_posterior_z_for(unsigned d) {
    // Leftover from CCMM: might still be relevant?
    // XXX: ratios of Dirichlet Processes won't really work; need to think
    // through the math a little bit more. The problem is that the original
    // clustering has a different number of clusters than the new one, and
    // aynway it doesn't make sense to loop over cluster pairs. Really we
    // need to integrate out the clustersings (marginalize over all possible
    // clusterings in both cases). But this is unfortunately prohibitively
    // expensive for DPMM.
    
    // Gibbs sampler for XCAT
    // _cluster [ zdn ] [ cdm ]

    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];
        unsigned old_zdn = _z[d][n];
        unsigned old_cdm = _c[d][old_zdn];

        _cluster[old_zdn][old_cdm].remove_no_ndsum(w,d);
        _cluster_marginal[old_zdn].remove_no_ndsum(w,d);
        
        vector<double> lp_z_dn;
        for (int m = 0; m < FLAGS_M; m++) {

            unsigned test_cdm = _c[d][m];
            // XXX: _nd[d] is this the correct model? should we be subsetting on
            // the stuff available in this view
            if (is_cluster_marginal) {
                lp_z_dn.push_back(log(_eta[w] + _cluster_marginal[m].nw[w]) -
                        log(_eta_sum + _cluster_marginal[m].nwsum) +
                        log(FLAGS_cc_xi + _cluster_marginal[m].nd[d]) -
                        log(FLAGS_cc_xi*FLAGS_M + _nd[d]-1));
            } else {
                lp_z_dn.push_back(log(_eta[w] + _cluster[m][test_cdm].nw[w]) -
                        log(_eta_sum + _cluster[m][test_cdm].nwsum) +
                        log(FLAGS_cc_xi + _cluster[m][test_cdm].nd[d]) -
                        log(FLAGS_cc_xi*FLAGS_M + _nd[d]-1));
            }
        }
        // XXX: other components??
    
        // Update the assignment
        _z[d][n] = sample_unnormalized_log_multinomial(&lp_z_dn);
        
        unsigned new_zdn = _z[d][n];
        unsigned new_cdm = _c[d][_z[d][n]];

        _cluster[new_zdn][new_cdm].add_no_ndsum(w,d);
        _cluster_marginal[new_zdn].add_no_ndsum(w,d);

        if (new_zdn == old_zdn) {
            _m_failed += 1;
        } else {
            // LOG(INFO) << "MOVED!";
        }
        _m_proposed += 1;
    }
}

double SoftCrossCatMM::compute_log_likelihood() {
    // Compute the log likelihood for the tree
    double log_lik = 0;
    /*

    // TODO: is this really correct? it seems ok at least, but likelihood is
    // usually interpreted as model likelihood or data likelihood? p(m|x) or
    // p(x|m) it sort of doesn't matter
    //
    for (int d = 0; d < _D.size(); d++) {
        // google::dense_hash_map<unsigned, unsigned> collapsed_w;
        // collapsed_w.set_empty_key(kEmptyUnsignedKey);

        for (int n = 0; n < _D[d].size(); n++) {
            // LDA part
            // This is p(z|x)
            
            unsigned w = _D[d][n];
            unsigned zdn = _z[d][n];
            // collapsed_w[w] += 1;
            if (is_cluster_marginal) {
                // log_lik += log(_eta[w] + _cluster_marginal[zdn].nw[w]) -
                //         log(_eta_sum + _cluster_marginal[zdn].nwsum) +
                //         log(FLAGS_cc_xi + _cluster_marginal[zdn].nd[d]) -
                //         log(FLAGS_cc_xi*FLAGS_M + _nd[d]-1);
                log_lik += log(FLAGS_cc_xi + _cluster_marginal[zdn].nd[d]) -
                           log(FLAGS_cc_xi*FLAGS_M + _nd[d]-1);
            } else {
                unsigned cdm = _c[d][zdn];
                // log_lik += log(_eta[w] + _cluster[zdn][cdm].nw[w]) -
                //         log(_eta_sum + _cluster[zdn][cdm].nwsum) +
                //         log(FLAGS_cc_xi + _cluster[zdn][cdm].nd[d]) -
                //         log(FLAGS_cc_xi*FLAGS_M + _nd[d]-1);
                log_lik += log(FLAGS_cc_xi + _cluster[zdn][cdm].nd[d]) -
                           log(FLAGS_cc_xi*FLAGS_M + _nd[d]-1);
            }

            // Cluster part
            unsigned l = _c[d][zdn];
            log_lik += gammaln(_eta[w] + _cluster[zdn][l].nw[w])
                       - gammaln(_eta_sum + _cluster[zdn][l].nwsum);
        }
            
    }
    */

    // HACK: ll is just sum of moved percentages
    
    double cluster_move_p = 0;
    if (_c_proposed > 0) {
        cluster_move_p = 100 - _c_failed / (double)_c_proposed*100;
    }
    double view_move_p = 0;
    if (_m_proposed > 0) {
        view_move_p = 100 - _m_failed / (double)_m_proposed*100;
    }
    log_lik = -cluster_move_p - view_move_p;

    return log_lik;
}

string SoftCrossCatMM::current_state() {
    _output_filename = FLAGS_mm_datafile;
    _output_filename += StringPrintf("-alpha%f-eta%f-xi%f",
            FLAGS_mm_alpha,
            _eta_sum / (double)_eta.size(),
            FLAGS_cc_xi);

    // Compute a vector of the clsuter sizes for printing.
    vector<string> cluster_sizes;
    for (multiple_clustering::iterator c_itr = _cluster.begin();
        c_itr != _cluster.end();
        c_itr++) {
        unsigned m = c_itr->first;
        clustering& cm = c_itr->second;
        cluster_sizes.push_back(StringPrintf("[%d: c: %d f: %d]" , m, cm.size(), _cluster_marginal[m].nwsum));
    }

    return StringPrintf(
            "ll = %f (%f at %d) alpha = %f eta = %f xi = %f K = %s M = %d <cm: %d (%.3f%%)> <vm: %d (%.3f%%)>",
            _ll, _best_ll, _best_iter,
            FLAGS_mm_alpha,
            _eta_sum / (double)_eta.size(),
            FLAGS_cc_xi,
            JoinStrings(cluster_sizes, " ").c_str(),
            FLAGS_M,
            _c_proposed-_c_failed,
            100 - _c_failed / (double)_c_proposed*100,
            _m_proposed-_m_failed,
            100 - _m_failed / (double)_m_proposed*100);
}

void SoftCrossCatMM::resample_posterior() {
    CHECK_GT(_lV, 0);
    CHECK_GT(_lD, 0);
    CHECK_GT(_c.size(), 0);

    // Resample the cluster indicators
    _c_proposed = _c_failed = 0;
    _m_proposed = _m_failed = 0;
    for (int d = 0; d < _D.size(); d++) {
        if (FLAGS_M > 1) {
            resample_posterior_z_for(d);
        }
        // LOG(INFO) <<  "  resampling document " <<  d;
        // For each clustering (view), resample this document's cluster indicator
        resample_posterior_c_for(d);
    }

    // Compute the number of documents in this cluster with actual
    // features in the game
    map<unsigned, map<unsigned, unsigned> > nonzero_docs;
    unsigned non_zero_docs = 0;
    for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;  // = document number

        map<unsigned, map<unsigned, unsigned> > things_im_in;
        for (int n = 0; n < _D[d].size(); n++) {
            unsigned w = _D[d][n];
            things_im_in[_z[d][n]][_c[d][_z[d][n]]] += 1;
        }

        for (map<unsigned, map<unsigned, unsigned> >::iterator itr = things_im_in.begin();
                itr != things_im_in.end();
                itr++) {
            unsigned m = itr->first;
            for (map<unsigned,unsigned>::iterator itr2 = itr->second.begin();
                    itr2 != itr->second.end();
                    itr2++) {
                unsigned c = itr2->first;
                unsigned count = itr2->second;

                if (count > 0) {
                    nonzero_docs[m][c] += 1;
                }
            }
        }
    }

    // Write the current cluster sizes to the console
    for (multiple_clustering::iterator c_itr = _cluster.begin();
            c_itr != _cluster.end();
            c_itr++) { 
        unsigned m = c_itr->first;
        // Summarize what features are assigned to this view
        LOG(INFO) << "M[" << m << "] (f " << _cluster_marginal[m].nwsum << ") " << show_chopped_sorted_nw(_cluster_marginal[m].nw);

        // Show the contents of the clusters
        unsigned test_sum = 0;
        for (clustering::iterator itr = _cluster[m].begin();
                itr != _cluster[m].end();
                itr++) {
            unsigned l = itr->first;

            // LOG(INFO) << "  C[" << l << "] (d " << itr->second.ndsum << ") " 
            //           << " " << show_chopped_sorted_nw(itr->second.nw);
            LOG(INFO) << "  C[" << l << "] (d " << nonzero_docs[m][l] << "/" << itr->second.ndsum << " nw " << itr->second.nwsum << ") " 
                      << " " << show_chopped_sorted_nw(itr->second.nw);

            test_sum += _cluster[m][l].ndsum;
        }
        CHECK_EQ(test_sum,_lD);  // make sure we haven't lost any docs
    }
    LOG(INFO) << "||| cluster moves " << _c_proposed-_c_failed << " / " << _c_proposed << " " 
        << StringPrintf("(%.3f%%)", 100 - _c_failed / (double)_c_proposed*100)
        << " ||| view moves " << _m_proposed-_m_failed << " / " << _m_proposed << " " 
        << StringPrintf("(%.3f%%)", 100 - _m_failed / (double)_m_proposed*100);
}

// Write out all the data in an intermediate format
void SoftCrossCatMM::write_data(string prefix) {
    // string filename = StringPrintf("%s-%d-%s.hlda.bz2", get_base_name(_output_filename).c_str(), FLAGS_random_seed,
    //         prefix.c_str());
    string filename = StringPrintf("%s-%d-%s.hlda", get_base_name(_output_filename).c_str(),
            FLAGS_random_seed, prefix.c_str());
    VLOG(1) << "writing data to [" << filename << "]";

    // XXX: Order in which f_raw and filtering ostream are declared matters.
    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    // f = get_bz2_ostream(filename);

    f << current_state() << endl;
    f << _iter << "\t" << _best_iter << "\t" << _best_ll << endl;

    // Write out how each document gets clustered in each view.
    for (int d = 0; d < _D.size(); d++) {
        for (int n = 0; n < _D[d].size(); n++) {
            for (multiple_clustering::iterator c_itr = _cluster.begin();
                c_itr != _cluster.end();
                c_itr++) { 
                unsigned m = c_itr->first;
                clustering& cm = c_itr->second;
                
                if (_z[d][n] == m) {
                    f << d << "\t" << n << "\t" << _c[d][m] << "\t" << _z[d][n] << endl;
                }
            }
        }
    }
    f << "END" << endl;
    //boost::iostreams::flush(f);
    VLOG(1) << "done";
}

// Restore from the intermediate model
bool SoftCrossCatMM::restore_data_from_prefix(string prefix) {
    current_state();   // HACK
    return restore_data_from_file(
                StringPrintf("%s-%d-%s.hlda", get_base_name(_output_filename).c_str(), FLAGS_random_seed, prefix.c_str())
            );
}
bool SoftCrossCatMM::restore_data_from_file(string filename) {
    bool finished = false;
    struct stat stFileInfo; 
    ifstream input_file(filename.c_str(), ios_base::in | ios_base::binary);

    clean_initialization();

    // Attempt to get the file attributes 
    if(stat(filename.c_str(),&stFileInfo) == 0) {
        LOG(INFO) << "attempting to restore data from [" << filename << "]";

        ifstream f(filename.c_str(), ios_base::in | ios_base::binary);

        string curr_line;
        getline(input_file, curr_line);
        getline(input_file, curr_line);
        vector<string> tokens;
        SplitStringUsing(StringReplace(curr_line, "\n", "", true), "\t", &tokens);
        CHECK_EQ(tokens.size(), 3);
        _iter = atoi(tokens.at(0).c_str()) + 1;
        _best_iter = atoi(tokens.at(1).c_str());
        _best_ll = atof(tokens.at(2).c_str());

        while (true) {
            getline(input_file, curr_line);

            if (curr_line == "END") {
                LOG(INFO) << "read correctly, resuming from iter=" << _iter;
                finished = true;

                // Add the documents into the clustering
                for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
                    unsigned d = d_itr->first;  // = document number
                    for (int m = 0; m < FLAGS_M; m++) {
                        _cluster[m][_c[d][m]].ndsum += 1; // Can't use ADD b/c we need to maintain ndsum over all the views
                        _cluster_marginal[m].ndsum += 1; // Can't use ADD b/c we need to maintain ndsum over all the views
                    }
                }
                break;
            }
            if (input_file.eof()) {
                break;
            }

            vector<string> tokens;
            SplitStringUsing(StringReplace(curr_line, "\n", "", true), "\t", &tokens);
            CHECK_EQ(tokens.size(), 4);
            unsigned d = atoi(tokens.at(0).c_str());
            unsigned n = atoi(tokens.at(1).c_str());
            unsigned cdm = atoi(tokens.at(2).c_str());
            unsigned zdn = atoi(tokens.at(3).c_str());

            _z[d][n] = zdn;
            _c[d][zdn] = cdm;

            if (cdm >= _current_component[zdn]) {
                _current_component[zdn] = cdm + 1;
            }

            // Cant use ADD b/c we need to keep the ndsum consistent across the
            // views
            unsigned w = _D[d][n];
            _cluster[zdn][cdm].add_no_ndsum(w, d);
            _cluster_marginal[zdn].add_no_ndsum(w, d);
        }
        VLOG(1) << "done";
    }

    return finished;
}

