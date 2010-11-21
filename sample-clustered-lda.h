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

#ifndef SAMPLE_CLUSTERED_LDA_H_
#define SAMPLE_CLUSTERED_LDA_H_

#include <string>
#include <vector>

#include "ncrp-base.h"

// Number of topics
DECLARE_int32(T);

// Number of clusters
DECLARE_int32(K);

// Smoother on clusters
DECLARE_double(xi);

// File containing the words and their features
DECLARE_string(word_features_file);

// Should clustering be constrained to only guys that have the same type? I.e.
// whereas before we would draw our word-clusters across the entire word-feature
// space, now the clustering is type-dependent.
DECLARE_bool(type_dependent_clustering);

class Feature {
    public:
        Feature(unsigned feature_id, unsigned count) : _feature_id(feature_id), _count(count) { }

    public:
        unsigned _feature_id;
        unsigned _count;
};

class NestedDocument {
    public:
        class WordFeatures {
            public:
                WordFeatures(unsigned word_id, string word_name, unsigned word_type_id)
                    : _word_id(word_id), _word_name(word_name), _word_type_id(word_type_id), _topic_indicator(0) { }

                void uniform_initialization() {
                    _topic_indicator = sample_integer(FLAGS_T);
                    _cluster_indicator = sample_integer(FLAGS_K);
                }

            public:
                unsigned _word_id;
                unsigned _word_type_id;
                string _word_name;

                unsigned _topic_indicator;
                unsigned _cluster_indicator;
        };

        NestedDocument(string doc_name, unsigned doc_id) : _doc_name(doc_name), _doc_id(doc_id) { }

    public:
        string _doc_name;
        unsigned _doc_id;
        vector<WordFeatures> _words;
};

typedef google::dense_hash_map<unsigned, vector<Feature> > WordIDToFeatures;

// Clustered Topic model with dirichlet-multinomial likelihood
class ClusteredLDA : public NCRPBase {
    public:
        ClusteredLDA() { }
        
        // Set up initial assignments and load the doc->word and word->feature maps
        void initialize();

        // Write out a static dictionary required for decoding Gibbs samples
        void write_dictionary();

        // Write out the Gibbs sample
        void write_data(string prefix);

        double compute_log_likelihood();
    protected:
        // Load a document -> word file
        void load_documents(const string& filename);

        // Load a word -> features file
        void load_words(const string& filename);

        void resample_posterior();
        void resample_posterior_z_for(unsigned d, bool remove);
        void resample_posterior_w_for(unsigned d);

        string current_state();

    protected:
        vector<NestedDocument> _D;
        vector<CRP> _master_cluster;
        google::dense_hash_map<unsigned, google::sparse_hash_map<unsigned, CRP*> > _cluster;
        vector<CRP> _topic;

        unsigned _unique_type_count;

        unsigned _unique_feature_count;
        unsigned _total_feature_count;

        WordCode _feature_id_to_name;  // uniqe_id to string
        google::dense_hash_map<string, unsigned> _feature_name_to_id;

        WordCode _type_id_to_name;  // uniqe_id to string
        google::dense_hash_map<string, unsigned> _type_name_to_id;
        google::dense_hash_map<unsigned, unsigned> _word_id_to_type_id;

        WordIDToFeatures _features;

    
        vector<double> _xi; 
        double _xi_sum;

        // Base names of the flags
        string _word_features_moniker;
        string _datafile_moniker;
};

#endif  // SAMPLE_CLUSTERED_LDA_H_
