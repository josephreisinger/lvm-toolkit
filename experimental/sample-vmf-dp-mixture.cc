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
// Neal's Algorithm 8 for vMF-DP mixture
// G0 = vMF(mu, kappa)
// c_k ~ DP(G0, alpha)
// phi_k ~ vMF(c_k, xi)


#include <string>
#include <fstream>
#include <algorithm>

#include "sample-vmf-dp-mixture.h"


#define BOUNDLO(x) (((x)<FLAGS_epsilon_value)?(FLAGS_epsilon_value):(x))
#define BOUNDHI(x) (((x)>(1-FLAGS_epsilon_value))?(1-FLAGS_epsilon_value):(x))
#define BOUND(x) (BOUNDLO(BOUNDHI(x)))
#define BOUNDPROB(x) (((x)<-300)?(-300):((((x)>300)?(300):(x))))

using namespace boost::numeric;

const string kDirichletProcess = "dirichlet-process";
const string kDirichletMixture = "dirichlet";
const string kUniformMixture = "uniform";


const unsigned kM = 4;

// The number of MH iterations to run for.
DEFINE_int32(iterations,
             99999,
             "Number of MH sampling iterations to run.");

// Initial number of clusters
DEFINE_int32(K_initial,
             10,
             "Number of clusters initially.");

// The input data set. Consists of one document per line, with words separated
// by tabs, and appended with their counts. The first word in each document is
// taken as the document name.
DEFINE_string(datafile,
              "",
              "the input data set, words arranged in documents");

// Alpha controls the topic smoothing, with higher alpha causing more "uniform"
// distributions over topics. 
DEFINE_double(alpha,
              1.0,
              "Topic smoothing.");

// Kappa controls the width of the vMF clusters.
DEFINE_double(kappa,
              100,
              "vMF cluster concentration.");

// xi is the corpus concentration
DEFINE_double(xi,
              0.001,
              "Corpus mean concentration parameter.");

// How many steps to sample before we tune the MH random walk probabilities
DEFINE_int32(tune_interval,
             100,
             "how many MH iterations to perform before tuning proposals");

// Restricts the topics (phis) to live on the positive orthant by chopping all
// negative entries in the vector (sparsifying).
DEFINE_bool(restrict_topics,
            false,
            "if enabled, chops out negative entries in the topics");

// A unique identifier to add to the output moniker (useful for condor)
DEFINE_string(my_id,
             "null",
             "a unique identifier to add to the output file moniker");

// Returns a string summarizing the current state of the sampler.
string vMFDPMixture::current_state() {
    return StringPrintf("ll = %f (%f at %d) xi=%.3f || accept %: phi %.3f", _ll, _best_ll,
            _best_iter, _xi, 1.0-_rejected_phi/(double)_proposed_phi);
}

void vMFDPMixture::initialize() {
    LOG(INFO) << "initialize";
    LOG(INFO) << "Neal's Algorithm 8 nonconjugate DP sampler using m=" << kM;

    _best_ll = 0;
    _best_iter = 0;

    _K = FLAGS_K_initial;

    _filename = FLAGS_datafile;

    // Set up the mean hyperparameter
    _mu.resize(_lV);
    for (int i = 0; i < _lV; i++) {
        _mu[i] = 1.0; 
    }
    _mu /= norm_2(_mu);
    CHECK_LT(fabs(1.0-norm_2(_mu)), FLAGS_epsilon_value);

    // XXX: Maybe better to do this
    // Topic distribution mean 
    // _mu = sample_spherical_gaussian(_mu_mean, 1.0/_mu_kappa);
    // CHECK_LT(fabs(1.0-norm_2(_mu)), epsilonValue);

    // Set up the per topic phis
    for (int t = 0; t < _K; t++) {
        // Originally we sampled this from a vMF, but its hellishly slow
        // _phi[t] = sample_vmf(_mu, _xi);
        // _phi[t] = sample_spherical_gaussian(_mu, _xi);
        // CHECK_LT(fabs(1.0-norm_2(_phi[t])), epsilonValue);
        _phi[t] = propose_new_phi(sample_spherical_gaussian(_mu, _xi));
        CHECK_LT(fabs(1.0-norm_2(_phi[t])), FLAGS_epsilon_value);
    } 
    _proposal_variance_phi = 1.0;
    _rejected_phi = 0;


    // Initialize the DP part
    _current_component = _K;

    for (int l = 0; l < _K; l++) {
        _c.insert(pair<unsigned,CRP>(l, CRP()));
    }

    // For each document, allocate a topic path for it there are several ways to
    // do this, e.g. a single linear chain, incremental conditional sampling and
    // random tree
    for (int d = 0; d < _lD; d++) {
        // set a random topic assignment for this guy
        _z[d] = sample_integer(_K);
        _c[_z[d]].nd[0]    += 1;  // # of words in doc d with topic z

        // resample_posterior_z_for(d);
    }

    // Cull the DP assignments
    for (clustering::iterator itr = _c.begin();
            itr != _c.end();
            itr++) {
        unsigned l = itr->first;
        if (itr->second.nd[0] == 0) {  // empty component
            _c.erase(itr);
            _phi.erase(l);
        }
    }

    // DCHECK(tree_is_consistent());
    _ll = compute_log_likelihood();

    _iteration = 0;
}

// This is the main workhorse function, calling each of the node samplers in
// turn.
void vMFDPMixture::resample_posterior() {
    for (int d = 0; d < _lD; d++) {
        resample_posterior_c(d);
    }
    for (vmf_clustering::iterator itr = _phi.begin();
            itr != _phi.end();
            itr++) {
        unsigned l = itr->first;

        VLOG(1) << "resampling phi_" << l << "...";
        resample_posterior_phi(l);
    }
    LOG(INFO) << "total clusters = " << _c.size(); 
    
    print_clustering_summary();

}

// Prints out the top few features from each cluster
void  vMFDPMixture::print_clustering_summary() {
    for (clustering::iterator itr = _c.begin();
            itr != _c.end();
            itr++) {
        unsigned l = itr->first;
        ublas::vector<double> doc_sum;
        doc_sum.resize(_lV);
        for (int d = 0; d < _lD; d++) {
            if (_z[d] == l) {
                for (int k = 0; k < _lV; k++) {
                    if (_v[d][k] > 0) {
                        doc_sum[k] += 1;
                    }
                }
            }
        }
        
        // Convert the ublas vector into a vector of pairs for sorting
        vector<word_score> sorted;
        for (int k = 0; k < _lV; k++) {
            if (doc_sum[k] > 0) {
                sorted.push_back(make_pair(_word_id_to_name[k], doc_sum[k]));
            }
        }

        sort(sorted.begin(), sorted.end(), word_score_comp);
        
        // Finally print out the summary
        string buffer = "";
        for (int k = 0; k < min((int)sorted.size(), 5); k++) {
            buffer += StringPrintf("%s %d ", sorted[k].first.c_str(), sorted[k].second);
        }

        LOG(INFO) << "_c[" << l << "] (size " << _c[l].nd[0] << ") nzf = " 
            << sorted.size() << " " << buffer;


    }
}

// Get new cluster assignments
void vMFDPMixture::resample_posterior_c(unsigned d) {
    unsigned old_assignment = _z[d];

    vector<pair<unsigned,double> > lp_z_d;

    for (google::dense_hash_map<unsigned,CRP>::iterator itr = _c.begin();
            itr != _c.end();
            itr++) {
        unsigned l = itr->first;

        unsigned top = _c[l].nd[0];
        if (_z[d] == l) {
            top -= 1;
        }

        // First add in the prior over the clusters, Neal's algorithm 8
        double sum = log(top) - log(_lD - 1 + FLAGS_alpha)
                    + logp_vmf(_v[d], _phi[l], FLAGS_kappa, true);

        lp_z_d.push_back(pair<unsigned,double>(l, sum));
    }

    // Add some additional new components if DP
    vmf_clustering new_phi;
    new_phi.set_empty_key(kEmptyUnsignedKey);
    unsigned temp_current_component = _current_component;
    for (int m = 0; m < kM; m++) {
        new_phi[temp_current_component] = propose_new_phi(sample_spherical_gaussian(_mu, _xi));
        double sum = log(FLAGS_alpha/(double)kM) - log(_lD - 1 + FLAGS_alpha)
                    + logp_vmf(_v[d], new_phi[temp_current_component], FLAGS_kappa, true);
        lp_z_d.push_back(pair<unsigned,double>(temp_current_component, sum));
        temp_current_component += 1;
    }            


    // Update the assignment
    _z[d] = sample_unnormalized_log_multinomial(&lp_z_d);
    _c[_z[d]].nd[0] += 1;
    _c[old_assignment].nd[0] -= 1;
    VLOG(1) << "resampling posterior z for " << d << ": " << old_assignment << "->" << _z[d];


    // Add the new one if necessary
    if (_z[d] >= _current_component) {
        _phi[_z[d]] = new_phi[_z[d]];
        _current_component = _z[d] + 1;
        _K += 1;
    }

    // Clean up for the DPMM
    if (_c[old_assignment].nd[0] == 0) {  // empty component
        _c.erase(old_assignment);
        _phi.erase(old_assignment);
        _K -= 1;
    }
}

// The resampling routines below are all basically boilerplate copies of
// each other. Probably should templatize this or something 
//
// The basic structure is:
//   (1) calculate the probability for the current setting
//   (2) sample a new setting from the proposal distribution
//   (3) calculate the new probability
//   (4) test the ratio
//   (5) reject by returning to the previous value, if necessary

void vMFDPMixture::resample_posterior_phi(unsigned index) {
    double logp_orig = logp_phi(index, false) + log_likelihood_phi(index, false);
    ublas::vector<double> old_phi(_phi[index]);  // Check copy semantics

    VLOG(1) << "proposing new phi_" << index;

    _phi[index] = propose_new_phi(_phi[index]);
    double logp_new = logp_phi(index, false) + log_likelihood_phi(index, false);

    if (log(sample_uniform()) > logp_new - logp_orig) {  // reject
        VLOG(1) << "... rejected.";

        _phi[index] = old_phi;
        _rejected_phi += 1;
    }
    _proposed_phi += 1;
}

// Computes the log posterior likelihood, e.g. the probability of observing the
// documents that we do, given the model parameters.
double vMFDPMixture::compute_log_likelihood() {
    double ll = 0;
    for (int i = 0; i < _lD; i++) {
        ll += logp_v(i, true);
    }
    LOG(INFO) << ll;

    ll += logp_dirichlet_process(_c, FLAGS_alpha);
    LOG(INFO) << logp_dirichlet_process(_c, FLAGS_alpha);
    return ll;
}

// Log likelihoods for internal nodes are computed as the sum of the children's logp's, given 
// the setting for the internal node.

double vMFDPMixture::log_likelihood_phi(unsigned index, bool normalize) {
    // phi's children are the v's
    double ll = 0;
    for (int d = 0; d < _lD; d++) {
        ll += logp_v(d, normalize);
    }
    return ll;
}

// Direct log probabilities of assignments
//
double vMFDPMixture::logp_phi(unsigned index, bool normalize) {
    VLOG(1) << "computing logp of phi_" << index;
    return logp_vmf(_phi[index], _mu, _xi, normalize);
    // return logp_vmf(_phi.at(index), _alpha);
}

double vMFDPMixture::logp_v(unsigned index, bool normalize) {
    double result = 0;
    result = logp_vmf(_v[index], _phi[_z[index]], FLAGS_kappa, normalize);

    VLOG(2) << "    logp_v_" << index << " = " << result;
    return result;
}

// Proposal distributions

// Phi is calculated by drawing from a spherical gaussian and then mapping
// it onto the hypersphere.
ublas::vector<double> vMFDPMixture::propose_new_phi(const ublas::vector<double>& phi) {
    ublas::vector<double> result = phi + sample_gaussian_vector(0, _proposal_variance_phi, phi.size());

    // Force the draws onto the positive orthant by setting all negative entires
    // to zero
    if (FLAGS_restrict_topics) {
        for (int i = 0; i < result.size(); i++) {
            if (result[i] < 0) {
                result[i] = 0;
            }
        }
    }

    return result / norm_2(result);
}

double vMFDPMixture::get_new_proposal_variance(string var, double current, double reject_rate) {
    double accept_rate = 1.0 - reject_rate;

    LOG(INFO) << StringPrintf("TUNING: original variance: %s=%f", var.c_str(), current);

    // To handle our mu blowing up
    if (current > 10000) { 
        current = 10000;
    }
    // This voodoo is pulled directly from pymc
    if (accept_rate < 0.001) {
        current *= 0.1;  // reduce by 90 percent
    } else if (accept_rate < 0.05) {
        current *= 0.5;  // reduce by 50 percent
    } else if (accept_rate < 0.2) {
        current *= 0.9;  // reduce by ten percent
    } else if (accept_rate > 0.95) {
        current *= 10.0;  // increase by factor of ten
    } else if (accept_rate > 0.75) {
        current *= 2.0;  // increase by double
    } else if (accept_rate > 0.5) {
        current *= 1.1;  // increase by ten percent
    }
    LOG(INFO) << StringPrintf("TUNING: new variance: %s=%f", var.c_str(), current);
    return current;
}

// Tune the proposal distributions
void vMFDPMixture::tune() {
    _proposal_variance_phi = get_new_proposal_variance("phi", _proposal_variance_phi, _rejected_phi / (double)_proposed_phi);

    // Reset the reject counts to zero
    _rejected_phi = 0;
    _proposed_phi = 0;
}
            
// Actually does all the sampling and takes care of accounting stuff
void vMFDPMixture::run(int iterations) {
    LOG(INFO) << "begin sampling...";
    bool found_first_best = false;  // HACK for getting nan on the first round
    for (; _iteration < iterations; _iteration++) {
        if (_iteration > 0 && _iteration % FLAGS_tune_interval == 0) {
            tune();
        }

        resample_posterior();

        _ll = compute_log_likelihood();

        if (!isnan(_ll) && (_ll > _best_ll || !found_first_best)) {
            found_first_best = true;
            _best_ll = _ll;
            _best_iter = _iteration;

            LOG(INFO) << "Resampling iter = " << _iteration << " " << current_state() << " *";

            write_data("best");
        } else {
            LOG(INFO) << "Resampling iter = " << _iteration << " " << current_state();
        }

        write_data(StringPrintf("last", _iteration));
            

        if (_iteration % FLAGS_sample_lag == 0 && FLAGS_sample_lag > 0) {
            write_data(StringPrintf("sample-%05d", _iteration));
        }
    }
}


void vMFDPMixture::load_data(const string& input_file_name) {
    LOG(INFO) << "Loading document " << input_file_name;

    _lD = 0;
    unsigned unique_word_count = 0;

    map<string, unsigned> word_to_id;

    std::vector<ublas::compressed_vector<double> > temp_docs;

    _V.clear();
    _word_id_to_name.clear();

    CHECK_STRNE(FLAGS_datafile.c_str(), "");

    ifstream input_file(input_file_name.c_str());
    CHECK(input_file.is_open());

    string curr_line;
    while (true) {
        if (input_file.eof()) {
            break;
        }
        getline(input_file, curr_line);
        std::vector<string> words;
        curr_line = StringReplace(curr_line, "\n", "", true);

        SplitStringUsing(curr_line, "\t", &words);

        if (words.size() == 0) {
            continue;
        }

        // TODO(jsr) simplify this
        temp_docs.push_back(ublas::compressed_vector<double>(words.size()-1, words.size()-1));

        for (int i = 0; i < words.size(); i++) {
            CHECK_STRNE(words[i].c_str(), "");
            
            if (i == 0) {
                VLOG(1) << "found new document [" << words[i] << "]";
                continue;
            }

            VLOG(2) << words.at(i);

            std::vector<string> word_tokens;
            SplitStringUsing(words.at(i), ":", &word_tokens);
            CHECK_EQ(word_tokens.size(), 2);

            string word = word_tokens.at(0);
            double freq = atof(word_tokens.at(1).c_str());
            VLOG(2) << word << " " << freq;

            // Is this a new word?
            if (word_to_id.find(word) == word_to_id.end()) {
                word_to_id[word] = unique_word_count;
                unique_word_count += 1;
                _word_id_to_name[word_to_id[word]] = word;
                _V.insert(word_to_id[word]);
            }
            
            // This bit is pretty gross, truly; dynamically resize the sparse vector as needed
            // since there is no push_back or the equivalent
            if (temp_docs.at(_lD).size() <= word_to_id[word]) {
                temp_docs.at(_lD).resize(word_to_id[word]+1, true);
            } 

            // TODO(jsr) this forces us to use each word only once per doc in the input file
            temp_docs.at(_lD).insert_element(word_to_id[word], freq);
        }
        CHECK_GT(sum(temp_docs.at(_lD)), 0);
        temp_docs.at(_lD) /= norm_2(temp_docs.at(_lD));  // L2 norm
                
        _lD += 1;
    }

    _lV = _V.size();

    // Copy the temp docs over
    for (int d = 0; d < _lD; d++) {
        _v[d].resize(_lV);
        for (int k = 0; k < temp_docs[d].size(); k++) {
            _v[d][k] = temp_docs[d][k];
        }
    }

    LOG(INFO) << "Loaded " << _lD << " documents with "
        << _V.size() << " unique words from "
        << input_file_name;
}


// Write out all the data in an intermediate format
void vMFDPMixture::write_data(string prefix) {
    string filename = StringPrintf("%s-%s-%s.params", _filename.c_str(), FLAGS_my_id.c_str(), prefix.c_str());

    ofstream f;
    f.open(filename.c_str(), ios::out);

    f << current_state() << endl;

    f << "iteration " << _iteration << endl;
    f << "best_ll " << _best_ll << endl;
    f << "best_iter " << _best_iter << endl;;
    f << "proposal_variance_phi " << _proposal_variance_phi << endl;

    f << "vocab "; 
    for (WordCode::iterator itr = _word_id_to_name.begin(); itr != _word_id_to_name.end(); itr++) {
        f << itr->first << ":" << itr->second << "\t";
    }
    f << endl;

    f << "alpha " << FLAGS_alpha << endl;
    f << "kappa " << FLAGS_kappa << endl;
    f << "xi " << FLAGS_xi << endl;
    
    for (vmf_clustering::iterator itr = _phi.begin();
            itr != _phi.end();
            itr++) {
        unsigned l = itr->first;

        f << "phi_" << l << " " << _phi[l] << endl;
    }
    for (clustering::iterator itr = _c.begin();
            itr != _c.end();
            itr++) {
        unsigned l = itr->first;

        f << "c_" << l << " " << _c[l].nd[0] << endl;
    }
    for (int k = 0; k < _lD; k++) {
        f << "z_" << k << " " << _z[k] << endl;
    }

    f.close();
}


// Generate a vector of samples from the same gaussian
ublas::vector<double> sample_gaussian_vector(double mu, double si2, size_t dim) {
    ublas::vector<double> result(dim);
    for (int i = 0; i < dim; i++) {
        result[i] = si2*sample_gaussian()+mu;
    }
    return result;
}

// Sample from a Gaussian (spherical) and then normalize the resulting draw onto
// the unit hypersphere
template <typename vec_t>
ublas::vector<double> sample_spherical_gaussian(const vec_t& mean, double si2) {
    ublas::vector<double> result(mean.size());
    for (int i = 0; i < mean.size(); i++) {
        result[i] = si2*sample_gaussian()+mean[i];
    }
    return result / norm_2(result);
}


// For now always assume mu and v can differ in underlying vector representation
template <typename vec_t1, typename vec_t2>
double logp_vmf(const vec_t1& v, const vec_t2& mu, double kappa, bool normalize) {
    CHECK_EQ(v.size(), mu.size());
    // TODO(jsr) figure out how to do this properly
    DCHECK_LT(fabs(1.0-norm_2(v)), FLAGS_epsilon_value);
    DCHECK_LT(fabs(1.0-norm_2(mu)), FLAGS_epsilon_value);

    unsigned p = v.size();

    if (normalize) {
        // Compute an approximate log modified bessel function of the first kind
        double l_bessel = approx_log_iv(p / 2.0 - 1.0, kappa);

        return kappa * inner_prod(mu, v) 
               + (p/2.0)*log(kappa / (2*M_PI)) 
               - log(kappa) 
               - l_bessel;
    } else {  // in the unnormalized case, don't compute terms only involving kappa
        return kappa * inner_prod(mu, v);
    }
}
// Computes the Abramowitz and Stegum approximation to the log modified bessel
// funtion of the first kind -- stable for high values of nu. See Chris Elkan's
double approx_log_iv(double nu, double z) {
    double alpha = 1 + pow(z / nu, 2);
    double eta = sqrt(alpha) + log(z / nu) - log(1+sqrt(alpha));
    return -log(sqrt(2*M_PI*nu)) + nu*eta - 0.25 * log(alpha);
}


// Computes the log prob of a value in a symmetric dirichlet
// or dirichlet process
double logp_dirichlet_process(clustering& value, double alpha) {
    unsigned dim = 0;
    double l;
    double s = 0;
    for (clustering::iterator itr = value.begin();
            itr != value.end();
            itr++) {
        s += itr->second.nd[0] - alpha;
        dim += 1;
    }

    // TODO: we shouldn't have to compute the sum
    double alpha_sum = alpha*((double)dim);
    l = gammaln(alpha_sum) - ((double)dim) * gammaln(alpha);

    for (clustering::iterator itr = value.begin();
            itr != value.end();
            itr++) {

        l += (alpha-1) * log(BOUND((itr->second.nd[0]-alpha)/s)); 
    }
    return BOUNDPROB(l);
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    init_random();

    vMFDPMixture h = vMFDPMixture(FLAGS_xi);
    h.load_data(FLAGS_datafile);
    h.initialize();

    h.run(FLAGS_iterations);
}



