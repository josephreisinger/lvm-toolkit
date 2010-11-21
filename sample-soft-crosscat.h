#ifndef SAMPLE_SOFT_CROSSCAT_MM_H_
#define SAMPLE_SOFT_CROSSCAT_MM_H_

#include <string>
#include <vector>
#include<map>

#include "gibbs-base.h"

// the number of clusterings (topics)
DECLARE_int32(M);

// the maximum number of clusters
DECLARE_int32(KMAX);

// Smoother on clustering
DECLARE_double(mm_alpha);

// Smoother on cross cat / topic model
DECLARE_double(cc_xi);

// File holding the data
DECLARE_string(mm_datafile);

// If toggled, the first view will be constrained to a single cluster
DECLARE_bool(cc_include_noise_view);

// Implements several kinds of mixture models (uniform prior, Dirichlet prior,
// DPCrossCatMM all with DP-Multinomial likelihood.
class SoftCrossCatMM : public GibbsSampler {
    public:
        SoftCrossCatMM() { }

        // Allocate all the documents at once (called for non-streaming)
        void batch_allocation();
        
        double compute_log_likelihood();

        void write_data(string prefix);
    protected:
        void resample_posterior();
        void resample_posterior_c_for(unsigned d);
        void resample_posterior_z_for(unsigned d);


        string current_state();

        double compute_log_likelihood_for(unsigned m, clustering& cm);

    protected:
        // Maps documents to clusters
        multiple_cluster_map _c; // Map [d][m] -> cluster_id
        multiple_cluster_map _z; // Map [d][n] -> view_id
        multiple_clustering _cluster;  // Map [m][z] -> chinese restaurant

        // Base names of the flags
        string _word_features_moniker;
        string _datafile_moniker;

        unsigned _ndsum;

        vector<unsigned> _current_component;  // # of clusters currently

        string _output_filename;

        // Count the number of feature movement proposals and the number that
        // fail
        unsigned _m_proposed;
        unsigned _m_failed;

        // Count cluster moves
        unsigned _c_proposed;
        unsigned _c_failed;
};

#endif  // SAMPLE_SOFT_CROSSCAT_MM_H_
