// Samples from a multinomial mixture with clutster proportion smoother xi
// and data likelihood smoother eta.
//
// Also builds in support for explicit feature selection via the prix-fixe topic
// model
//

#include <math.h>
#include <time.h>

#include "gibbs-base.h"
#include "sample-mm.h"

// the number of clusters
DEFINE_int32(K,
             50,
             "Number of clusters.");

// Prior over cluster sizes
DEFINE_string(mm_prior,
              "dirichlet",
              "Prior over clusters. Can be dirichlet, dirichlet-process, or uniform.");

// Smoother on clustering
DEFINE_double(mm_xi,
              1.0,
              "Smoother on the cluster assignments.");

// Smoother on cluster likelihood
DEFINE_double(mm_beta,
              1.0,
              "Smoother on the cluster likelihood.");

// Smoother on data/noise assignment
DEFINE_double(mm_alpha,
              1.0,
              "Smoother on the data/noise assignment.");

// Number of noise topics
DEFINE_int32(N,
             0,
             "Number of noise topics");

// File holding the data
DEFINE_string(mm_datafile,
              "",
              "Docify holding the data to be clustered.");

const string kDirichletProcess = "dirichlet-process";
const string kDirichletMixture = "dirichlet";
const string kUniformMixture = "uniform";

void MM::initialize() {
    LOG(INFO) << "initialize";

    CHECK(FLAGS_mm_prior == kDirichletMixture ||
          FLAGS_mm_prior == kDirichletProcess ||
          FLAGS_mm_prior == kUniformMixture);
    
    _current_component = FLAGS_K;

    _N = FLAGS_N+1;  // Total number of topic components

    // Initialize the per-topic dirichlet parameters
    // NOTE: in reality this would actually have to be /per topic/ as in one
    // parameter per node in the hierarchy. But since its not used for now its
    // ok.
    for (int l = 0; l < FLAGS_K; l++) {
        _xi.push_back(FLAGS_mm_xi);
    }
    _xi_sum = FLAGS_K*FLAGS_mm_xi;

    // Topic Smoother (there are N+1 topics, noise + signal)
    for (int l = 0; l < _N; l++) {
        _alpha.push_back(FLAGS_mm_alpha);
    }
    _alpha_sum = _N*FLAGS_mm_alpha;


    // Set up likelihood smoother
    for (int l = 0; l < _lV; l++) {
        _beta.push_back(FLAGS_mm_beta);
    }
    _beta_sum = FLAGS_mm_beta*_lV;

    // Set up clusters
    for (int l = 0; l < FLAGS_K; l++) {
        _phi.insert(pair<unsigned,CRP>(l, CRP()));
    }

    // Add each document to a cluster and optionally allocate some of its words
    // to the noise topics
    for (DocumentMap::iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;  // = document number

        // set a random cluster assignment for this guy
        _c[d] = sample_integer(FLAGS_K);

        // Initial word assignments
        for (int n = 0; n < d_itr->second.size(); n++) {
            unsigned w = d_itr->second[n];

            // Topic assignment
            _z[d][n] = sample_integer(_N);

            if (_z[d][n] == 0) {
                // test the initialization of maps
                CHECK(_phi[_c[d]].nw.find(w) != _phi[_c[d]].nw.end()
                        || _phi[_c[d]].nw[w] == 0);

                _phi[_c[d]].add(w, d);
            } else {
                _phi_noise[_z[d][n]].add(w, d);
            }
        }
        if (d > 0 && FLAGS_mm_prior != kDirichletProcess) {  // interacts badly with DP, at least asserts do
            CHECK((_phi.size() == FLAGS_K) || FLAGS_mm_prior == kDirichletProcess) << "doc " << d;
            resample_posterior_c_for(d);
            resample_posterior_z_for(d);
            CHECK((_phi.size() == FLAGS_K) || FLAGS_mm_prior == kDirichletProcess) << "doc " << d;
        }
    }

    // Cull the DP assignments
    if (FLAGS_mm_prior == kDirichletProcess) {
        for (google::dense_hash_map<unsigned,CRP>::iterator itr = _phi.begin();
                itr != _phi.end();
                itr++) {
            unsigned l = itr->first;
            if (itr->second.ndsum == 0) {  // empty component
                _phi.erase(itr);
                VLOG(1) << "erasing component";
            }
        }
    }

    _ll = compute_log_likelihood();
}

double MM::document_slice_log_likelihood(unsigned d, unsigned l) {
    // XXX: all of this stuff might be wrong basically; should be ratios of
    // gammas because we need to integrate over all possible orderings?
    double log_lik = 0; 
    CHECK((l < FLAGS_K) || FLAGS_mm_prior == kDirichletProcess);
    for (int n = 0; n < _D[d].size(); n++) {
        if (_z[d][n] == 0) {
            // likelihood of drawing this word
            unsigned w = _D[d][n];
            log_lik += log(_phi[l].nw[w]+_beta[w]) - log(_phi[l].nwsum+_beta_sum);

            // Topic model part
            if (FLAGS_N > 0) {
                log_lik += log(_alpha[0] + _phi[l].nd[d]) - log(_alpha_sum + _nd[d]);
            }
            // CHECK_LE(_phi[l].nd[d], _nd[d]);
            // CHECK_LE(_phi[l].nw[w], _phi[l].nwsum);
            // CHECK_LE(_alpha[0], _alpha_sum);
            // CHECK_LE(_beta[w], _beta_sum);
        }
    }
    return log_lik;
}

// Reallocate this document's words between the noise and signal topics
void MM::resample_posterior_z_for(unsigned d) {
    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];
        // Remove this document and word from the counts
        if (_z[d][n] == 0) {
            _phi[_c[d]].remove(w, d);
        } else {
            _phi_noise[_z[d][n]].remove(w, d);
        }

        vector<double> lp_z_dn;
        for (int l = 0; l < _N; l++) {
            if (l == 0) {
                // TODO: check the normalizer on documents here; do we normalize
                // to document length or to topic docs?

                lp_z_dn.push_back(log(_beta[w] + _phi[_c[d]].nw[w]) -
                        log(_beta_sum + _phi[_c[d]].nwsum) +
                        log(_alpha[0] + _phi[_c[d]].nd[d]) -
                        log(_alpha_sum + _nd[d]));
            } else {
                lp_z_dn.push_back(log(_beta[w] + _phi_noise[l].nw[w]) -
                        log(_beta_sum + _phi_noise[l].nwsum) +
                        log(_alpha[l] + _phi_noise[l].nd[d]) -
                        log(_alpha_sum + _nd[d]));
            }
        }

        // Update the assignment
        _z[d][n] = sample_unnormalized_log_multinomial(&lp_z_dn);

        // Update the counts
        if (_z[d][n] == 0) {
            _phi[_c[d]].add(w, d);
        } else {
            _phi_noise[_z[d][n]].add(w, d);
        }
    }
}

// Performs a single document's level assignment resample step
void MM::resample_posterior_c_for(unsigned d) {
    unsigned old_assignment = _c[d];
    CHECK_LT(d, _lD);

    // Remove document d from the clustering
    for (int n = 0; n < _D[d].size(); n++) {
        if (_z[d][n] == 0) {
            unsigned w = _D[d].at(n);
            _phi[_c[d]].remove(w,d);
        }
    }

    CHECK_LE(_phi[_c[d]].ndsum, _lD);

    vector<pair<unsigned,double> > lp_c_d;

    unsigned test_ndsum = 0;
    for (google::dense_hash_map<unsigned,CRP>::iterator itr = _phi.begin();
            itr != _phi.end();
            itr++) {
        unsigned l = itr->first;

        double log_lik = 0;
        
        // First add in the prior over the clusters
        if (FLAGS_mm_prior == kDirichletMixture) {
            log_lik += log(_xi[l] + _phi[l].ndsum) - log(_xi_sum + _lD);
        } else if (FLAGS_mm_prior == kDirichletProcess) {
            log_lik += log(_phi[l].ndsum) - log(_lD - 1 + FLAGS_mm_xi);
            test_ndsum += _phi[l].ndsum;
        }

        // Now account for the likelihood of the data (marginal posterior of
        // DP-Mult)
        CHECK((l < FLAGS_K) || FLAGS_mm_prior == kDirichletProcess) << "A";
        log_lik += document_slice_log_likelihood(d, l);

        lp_c_d.push_back(pair<unsigned,double>(l, log_lik));
    }
    // CHECK_EQ(test_ndsum, _lD-1);

    // Add an additional new component if DP
    if (FLAGS_mm_prior == kDirichletProcess) {
        double log_lik = log(FLAGS_mm_xi) - log(_lD - 1 + FLAGS_mm_xi);
        for (int n = 0; n < _D[d].size(); n++) {
            if (_z[d][n] == 0) {
                unsigned w = _D[d][n];
                log_lik += log(_beta[w]) - log(_beta_sum);
            }
        }

        lp_c_d.push_back(pair<unsigned,double>(_current_component, log_lik));
    }

    // Update the assignment
    _c[d] = sample_unnormalized_log_multinomial(&lp_c_d);
    VLOG(1) << "resampling posterior c for " << d << ": " << old_assignment << "->" << _c[d];


    // Add document d back to the clustering at the new assignment
    for (int n = 0; n < _D[d].size(); n++) {
        if (_z[d][n] == 0) {
            unsigned w = _D[d].at(n);
            _phi[_c[d]].add(w,d);
        }
    }

    CHECK_LE(_phi[_c[d]].ndsum, _lD);

    // Clean up for the DPMM
    if (FLAGS_mm_prior == kDirichletProcess) {
        if (_phi[old_assignment].ndsum == 0) {  // empty component
            _phi.erase(old_assignment);
        }
        // Make room for a new component if we selected the new one
        if (_c[d] == _current_component) {
            _current_component += 1;
        }
    }
}


double MM::compute_log_likelihood() {
    // Compute the log likelihood for the tree
    double log_lik = 0;

    // Compute the log likelihood of the level assignments (correctly?)
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        // Add in prior over clusters
        if (FLAGS_mm_prior == kDirichletMixture) {
            log_lik += log(_phi[_c[d]].ndsum+_xi[_c[d]]) - log(_lD +_xi_sum);
            CHECK_LT(_c[d], FLAGS_K);
            CHECK_LE(_phi[_c[d]].ndsum, _lD);
            CHECK_LE(_xi[_c[d]], _xi_sum);
        } else if (FLAGS_mm_prior == kDirichletProcess) {
            log_lik += log(_phi[_c[d]].ndsum) - log(_lD - 1 + FLAGS_mm_xi);
        }

        // Log-likelihood of the slice of the document accounted for by _c[d]
        CHECK_LE(log_lik, 0) << "hello3";
        CHECK((_c[d] < FLAGS_K) || FLAGS_mm_prior == kDirichletProcess) << "B";
        log_lik += document_slice_log_likelihood(d, _c[d]);
        CHECK_LE(log_lik, 0) << "hello2";

        // Account for the noise assignments
        for (int n = 0; n < _D[d].size(); n++) {
            if (_z[d][n] > 0) {
                // likelihood of drawing this word
                unsigned w = _D[d][n];
                log_lik += log(_phi_noise[_z[d][n]].nw[w]+_beta[w]) - log(_phi_noise[_z[d][n]].nwsum+_beta_sum)
                         + log(_alpha[_z[d][n]] + _phi_noise[_z[d][n]].nd[d]) - log(_alpha_sum + _nd[d]);
                CHECK_LE(_phi_noise[_z[d][n]].nd[d], _nd[d]);
            }
        }
        CHECK_LE(log_lik, 0) << "hello";


    }
    return log_lik;
}

string MM::current_state() {
  _output_filename = FLAGS_mm_datafile;
  _output_filename += StringPrintf("-xi%f-beta%f",
                            _xi_sum / (double)_xi.size(),
                            _beta_sum / (double)_beta.size());

  return StringPrintf(
      "ll = %f (%f at %d) xi = %f beta = %f alpha = %f K = %d N = %d",
      _ll, _best_ll, _best_iter,
      _xi_sum / (double)_xi.size(),
      _beta_sum / (double)_beta.size(), 
      _alpha_sum / (double)_alpha.size(), 
      _phi.size(), FLAGS_N);
}

void MM::resample_posterior() {
    CHECK_GT(_lV, 0);
    CHECK_GT(_lD, 0);
    CHECK_GT(_phi.size(), 0);

    // Interleaved version
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        VLOG(1) <<  "  resampling document " <<  d;
        CHECK((_phi.size() == FLAGS_K) || FLAGS_mm_prior == kDirichletProcess);
        resample_posterior_c_for(d);
        resample_posterior_z_for(d);
    }

    print_cluster_summary();
    print_noise_summary();
}

// Write out all the data in an intermediate format
void MM::write_data(string prefix) {
    // string filename = StringPrintf("%s-%d-%s.hlda.bz2", get_base_name(_output_filename).c_str(), FLAGS_random_seed,
    //         prefix.c_str());
    string filename = StringPrintf("%s-%s-N%d-%d-%s.hlda", get_base_name(_output_filename).c_str(),
            FLAGS_mm_prior.c_str(), FLAGS_N, FLAGS_random_seed,
            prefix.c_str());
    VLOG(1) << "writing data to [" << filename << "]";

    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    // get_bz2_ostream(filename, f);

    f << current_state() << endl;


    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        f << _c[d] << "\t" << _document_name[d] << endl;
    }

    f << "NOISE ASSIGNMENT" << endl;
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        f << _document_name[d];
        for (int n = 0; n < _D[d].size(); n++) {
            unsigned w = _D[d][n];
            f << "\t" << _word_id_to_name[w] << ":" << _z[d][n];
        }
        f << endl;
    }
}


// Prints out the top few features from each cluster
void  MM::print_noise_summary() {
    for (clustering::iterator itr = _phi_noise.begin();
            itr != _phi_noise.end();
            itr++) {
        unsigned l = itr->first;

        if (l == 0) {
            CHECK_EQ(itr->second.nwsum, 0);
        } else {
            string buffer = show_chopped_sorted_nw(itr->second.nw);
            
            // Convert the ublas vector into a vector of pairs for sorting
            LOG(INFO) << "N[" << l << "] (" << StringPrintf("%.3f\%", itr->second.nwsum / (double)_total_word_count)  << ") " << " " << buffer;
        }
    }
}

// Prints out the top few features from each cluster
void  MM::print_cluster_summary() {
    // Write the current cluster sizes to the console
    string s;
    for (google::dense_hash_map<unsigned,CRP>::iterator itr = _phi.begin();
            itr != _phi.end();
            itr++) {
        unsigned l = itr->first;
        s += StringPrintf("%d:%d ", l, _phi[l].ndsum);
    }
    LOG(INFO) << s;

    // Show the contents of the clusters
    for (clustering::iterator itr = _phi.begin();
            itr != _phi.end();
            itr++) {
        unsigned l = itr->first;

        string buffer = show_chopped_sorted_nw(itr->second.nw);
        
        // Convert the ublas vector into a vector of pairs for sorting
        LOG(INFO) << "C[" << l << "] (d " << itr->second.ndsum << ") " << " " << buffer;
    }
}
