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
#ifndef SAMPLE_SOFT_CROSSCAT_MM_H_
#define SAMPLE_SOFT_CROSSCAT_MM_H_

#include <string>
#include <vector>
#include<map>

#include "gibbs-base.h"

// Two main implementations:
//   (1) normal: treat the topic model part for a document as the set of
//       clusters picked out by the document (one per view)
//   (2) marginal: treat each "topic" as the marginal over all clusters inside
//       it; this model seems to make more sense to me
DECLARE_string(implementation);

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

// If toggled, will resume from the best model written so far
DECLARE_bool(cc_resume_from_best);

// Implements several kinds of mixture models (uniform prior, Dirichlet prior,
// DPCrossCatMM all with DP-Multinomial likelihood.
class SoftCrossCatMM : public GibbsSampler {
    public:
        SoftCrossCatMM() { }

        // Allocate all the documents at once (called for non-streaming)
        void batch_allocation();

        // Initialize the model cleanly
        void clean_initialization();
        
        double compute_log_likelihood();

        void write_data(string prefix);

        // Restore from the intermediate model
        bool restore_data(string prefix);
    protected:
        void resample_posterior();
        void resample_posterior_c_for(unsigned d);
        void resample_posterior_z_for(unsigned d);


        string current_state();

    protected:
        // Maps documents to clusters
        multiple_cluster_map _c; // Map [d][m] -> cluster_id
        multiple_cluster_map _z; // Map [d][n] -> view_id
        multiple_clustering _cluster;  // Map [m][z] -> chinese restaurant
        clustering _cluster_marginal;  // Map [m] -> chinese restaurant (marginal realization of _cluster)

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

        // Cluster marginal model?
        bool is_cluster_marginal;
};

#endif  // SAMPLE_SOFT_CROSSCAT_MM_H_
