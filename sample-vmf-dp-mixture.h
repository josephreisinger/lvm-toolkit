// Metropolis-Hastings sampler for an admixture of von Mises-Fisher distributions

#ifndef VMF_DP_MIXTURE_H_
#define VMF_DP_MIXTURE_H_

#include <iostream>
#include <fstream>
#include <string>

#include <math.h>

#include <set>
#include <map>
#include <vector>
#include <string>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "gibbs-base.h"
#include "strutil.h"

using namespace std;
using namespace boost::numeric;
using namespace google::protobuf;


// the number of iterations to run for.
DECLARE_int32(iterations);

// Initial number of clusters
DECLARE_int32(K_initial);

// The input data set. Consists of one document per line, with words separated
// by tabs, and appended with their counts. The first word in each document is
// taken as the document name.
DECLARE_string(datafile);

// Alpha controls the topic smoothing, with higher alpha causing more "uniform"
// distributions over topics. 
DECLARE_double(alpha);

// Kappa controls the width of the vMF clusters.
DECLARE_double(kappa);

// xi is the corpus concentration
DECLARE_double(xi);

// Restricts the topics (phis) to live on the positive orthant by chopping all
// negative entries in the vector (sparsifying).
DECLARE_bool(restrict_cluster_means);

// A unique identifier to add to the output moniker (useful for condor)
DECLARE_string(my_id);

typedef google::dense_hash_map< unsigned, ublas::vector<double> > vmf_data;
typedef google::dense_hash_map< unsigned, ublas::compressed_vector<double> > vmf_clustering;

class vMFDPMixture : public GibbsSampler {
    public:
        vMFDPMixture(double xi)
            : _xi(xi) { 
                _z.set_empty_key(kEmptyUnsignedKey);
                _c.set_empty_key(kEmptyUnsignedKey);
                _c.set_deleted_key(kDeletedUnsignedKey);
                _word_id_to_name.set_empty_key(kEmptyUnsignedKey);
                _v.set_empty_key(kEmptyUnsignedKey);
                _phi.set_empty_key(kEmptyUnsignedKey);
                _phi.set_deleted_key(kDeletedUnsignedKey);
            }
        virtual ~vMFDPMixture() { /* TODO: free memory! */ }

        // Set up all the data structures and initialize distributions to random values
        void initialize();

        // Run the MH sampler for iterations iterations. This is the main
        // workhorse loop.
        void run(int iterations);

        // Load a data file
        void load_data(const string& filename);
    
        // Dump the sufficient statistics of the model
        void write_data(string prefix);

        // return a string describing the current state of the hyperparameters
        // and log likelihood
        string current_state();

        // Prints out the top features of each cluster
        void print_clustering_summary();


    protected:
        // Performs an entire MH step, updating the posterior
        void resample_posterior();

        // The various component distributions
        void resample_posterior_phi(unsigned index);
        void resample_posterior_c(unsigned d);

        // Posterior likelihood of the data conditional on the model parameters
        double compute_log_likelihood();

        // Log-likelihood computation (sum of the log probabilities of the nodes children)
        double log_likelihood_phi(unsigned index, bool normalize);

        // Direct log probabilities of internal nodes given the settings of their parents
        double logp_phi(unsigned index, bool normalize);
        double logp_v(unsigned index, bool normalize);

        // Proposal distributions
        ublas::vector<double> propose_new_phi(const ublas::vector<double>& phi);

        // Tune the proposal steps
        double get_new_proposal_variance(string var, double current, double reject_rate);
        void tune();

    protected:
        cluster_map _z; // Map data point -> cluster
        clustering _c;  // Map [w][z] -> CRP

        // Parameters
        unsigned _K;  // Number of clusters

        unsigned _current_component;

        set<unsigned> _V;  // vocabulary
        unsigned _lV;  // size of vocab
        unsigned _lD;  // number of documents

        WordCode _word_id_to_name;  // uniqe_id to string

        double _ll;  // current log-likelihood
        double _best_ll;
        int    _best_iter;
        unsigned _iteration;  // Hold the global sampling step

        string _filename;  // output file name

        // Model hyperparameters
        ublas::vector<double> _mu;  // uniform mean vector constant
        double _xi;

        // Model parameters
        vmf_clustering _phi;

        // The actual observed documents (sparse matrix)
        vmf_data _v; 
        
        // Tallies for accounting and performing adaptive variance updates
        unsigned _rejected_xi;
        unsigned _rejected_phi;
        unsigned _proposed_phi;

        // Current proposal variance settings
        double _proposal_variance_xi;
        double _proposal_variance_phi;
};

ublas::vector<double> sample_gaussian_vector(double mean, double si2, unsigned dim);
template <typename vec_t>
ublas::vector<double> sample_spherical_gaussian(const vec_t& mean, double si2);

double approx_log_iv(double nu, double z);

// For now always assume mu and v can differ in underlying vector representation
template <typename vec_t1, typename vec_t2>
double logp_vmf(const vec_t1& v, const vec_t2& mu, double kappa, bool normalize);

double logp_dirichlet_process(clustering& value, double alpha);


#endif  // VMF_DP_MIXTURE_H_
