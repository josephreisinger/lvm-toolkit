// Basic implementation of clustered LDA with multinomial likelihood

#ifndef SAMPLE_CLUSTERED_LDA_H_
#define SAMPLE_CLUSTERED_LDA_H_

#include <string>
#include <vector>

#include "gibbs-base.h"

// Number of clusters
DECLARE_int32(K);

// Prior over cluster sizes
DECLARE_string(mm_prior);

// Smoother on clustering
DECLARE_double(mm_xi);

// Smoother on cluster likelihood
DECLARE_double(mm_beta);

// Smoother on data/noise assignment
DECLARE_double(mm_alpha);

// Number of noise topics
DECLARE_int32(N);

// File holding the data
DECLARE_string(mm_datafile);

// Implements several kinds of mixture models (uniform prior, Dirichlet prior,
// DPMM all with DP-Multinomial likelihood.
class MM : public GibbsSampler {
    public:
        MM() {
            _c.set_empty_key(kEmptyUnsignedKey);
            _z.set_empty_key(kEmptyUnsignedKey);
            _phi.set_empty_key(kEmptyUnsignedKey);
            _phi.set_deleted_key(kDeletedUnsignedKey);
            _phi_noise.set_empty_key(kEmptyUnsignedKey);
            _phi_noise.set_deleted_key(kDeletedUnsignedKey);
        }
        
        // Set up initial assignments and load the doc->word and word->feature maps
        void initialize();

        double compute_log_likelihood();

        void write_data(string prefix);
    protected:
        void resample_posterior();
        void resample_posterior_c_for(unsigned d);
        void resample_posterior_z_for(unsigned d);

        string current_state();

        double document_slice_log_likelihood(unsigned d, unsigned l);

        void print_cluster_summary();
        void print_noise_summary();
    protected:
        // Maps documents to clusters
        cluster_map _c; // Map data point -> cluster
        clustering _phi;  // Map [w][z] -> CRP

        topic_map _z; // Map word to noise or data
        clustering _phi_noise;  // Distribution for the noise

        vector<double> _beta; // Smoother for document likelihood
        double _beta_sum;

        vector<double> _alpha; // Smoother for noise / data assignment
        double _alpha_sum;

        vector<double> _xi; // Smoother for word likelihood
        double _xi_sum;

        unsigned _N;  // Total number of topic components (noise + 1)

        // Base names of the flags
        string _word_features_moniker;
        string _datafile_moniker;

        unsigned _ndsum;

        unsigned _current_component;  // # of clusters currently

        string _output_filename;
};

#endif  // SAMPLE_CLUSTERED_LDA_H_
