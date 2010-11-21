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
// Basic implementation of clustered LDA with multinomial likelihood

#ifndef SAMPLE_CROSSCAT_MM_H_
#define SAMPLE_CROSSCAT_MM_H_

#include <string>
#include <vector>
#include<map>

#include "gibbs-base.h"

// the number of feature clusters
DECLARE_int32(M);

// the maximum number of clusters
DECLARE_int32(KMAX);

// Smoother on clustering
DECLARE_double(mm_alpha);

// Smoother on cross cat clustering
DECLARE_double(cc_xi);

// File holding the data
DECLARE_string(mm_datafile);

// Number of feature moves to make
DECLARE_double(cc_feature_move_rate);

// If toggled, the first view will be constrained to a single cluster
DECLARE_bool(cc_include_noise_view);

// Basically controls whether and how we should do cross-cat on the features.
// Implemented using MH steps.
DECLARE_string(cross_cat_prior);

const string kCrossCatOff = "off";

// typedef google::dense_hash_map<unsigned, unsigned> collapsed_document;
typedef map<unsigned, unsigned> collapsed_document;
// typedef google::dense_hash_map<unsigned, collapsed_document> collapsed_document_collection;
typedef map<unsigned, collapsed_document> collapsed_document_collection;

// Implements several kinds of mixture models (uniform prior, Dirichlet prior,
// DPCrossCatMM all with DP-Multinomial likelihood.
class CrossCatMM : public GibbsSampler {
    public:
        CrossCatMM() { }

        // Allocate all the documents at once (called for non-streaming)
        void batch_allocation();
        
        double compute_log_likelihood();

        void write_data(string prefix);
    protected:
        void resample_posterior();
        void resample_posterior_z_for(unsigned d, unsigned m, bool remove);
        void resample_posterior_m(double percent);
        void resample_posterior_m_for(unsigned tw);


        string current_state();

        double compute_log_likelihood_for(unsigned m, clustering& cm);
        double cross_cat_clustering_log_likelihood(unsigned w, unsigned m);
        double cross_cat_reassign_features(unsigned old_m, unsigned new_m, unsigned w);

    protected:
        // Maps documents to clusters
        multiple_cluster_map _z; // Map (data point, clustering) -> cluster_id
        multiple_clustering _c;  // Map [w][z] -> CRP

        cluster_map _m;          // Map vocab -> cluster
        clustering _b;

        // Base names of the flags
        string _word_features_moniker;
        string _datafile_moniker;

        unsigned _ndsum;

        vector<unsigned> _current_component;  // # of clusters currently

        // Document clustering smoother
        vector<double> _alpha;
        double _alpha_sum;

        // Cross-cat smoother
        vector<double> _xi;
        double _xi_sum;

        string _output_filename;

        // Count the number of feature movement proposals and the number that
        // fail
        unsigned _m_proposed;
        unsigned _m_failed;

        // Count cluster moves
        unsigned _c_proposed;
        unsigned _c_failed;

        // Data structure to hold the documents optimzed for mixture model
        // computation (i.e. we don't need to break up each feature into
        // occurrences, and can instead treat the count directly)
        collapsed_document_collection _DD;
};

#endif  // SAMPLE_CROSSCAT_MM_H_
