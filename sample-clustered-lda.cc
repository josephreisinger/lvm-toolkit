#include <math.h>
#include <time.h>

#include "ncrp-base.h"
#include "sample-clustered-lda.h"

// the number of topics
DEFINE_int32(T,
             10,
             "Number of topics.");

// the number of clusters
DEFINE_int32(K,
             50,
             "Number of clusters.");

// smoother on the clusters
DEFINE_double(xi,
             0.1,
             "Cluster smoother.");

// File containing the words and their features
DEFINE_string(word_features_file,
              "",
              "File containig the words and their features.");

// Should clustering be constrained to only guys that have the same type? I.e.
// whereas before we would draw our word-clusters across the entire word-feature
// space, now the clustering is type-dependent.
DEFINE_bool(type_dependent_clustering,
            false,
            "Should the word clustering be done per type (true) or across all"
            "words (false).");

// Initialize the hLDABase tree to something
void ClusteredLDA::initialize() {
    LOG(INFO) << "initialize";

    // Incompatibilities
    // CHECK(!FLAGS_gem_sampler);
    // CHECK(!FLAGS_use_reject_option);
    //
    CHECK_STRNE(FLAGS_ncrp_datafile.c_str(), "");
    CHECK_STRNE(FLAGS_word_features_file.c_str(), "");

    // Get the base names of the datafile and word features file
    _datafile_moniker = get_base_name(FLAGS_ncrp_datafile);
    _word_features_moniker = get_base_name(FLAGS_word_features_file);

    _topic.clear();
    _master_cluster.clear();


    // Initialize the topic chain
    for (int t = 0; t < FLAGS_T; t++) {
        _topic.push_back(CRP(0,0));
        _topic[t].label = "NULL";
    }
    // Initialize the clusters (here for normal, otherwise we do it as we load
    // the words
    if (!FLAGS_type_dependent_clustering) {
        for (int k = 0; k < FLAGS_K; k++) {
            _master_cluster.push_back(CRP(0,0));
            _master_cluster[k].label = "NULL";
            _cluster[0][k] = &_master_cluster[k];
        }
    }

    // Initialize alpha
    _alpha.clear();
    for (int l = _alpha.size(); l < FLAGS_T; l++) {
        _alpha.push_back(FLAGS_ncrp_alpha);
    }
    _alpha_sum = FLAGS_T*FLAGS_ncrp_alpha;

    // Initialize xi
    _xi.clear();
    for (int l = _xi.size(); l < FLAGS_K; l++) {
        _xi.push_back(FLAGS_xi);
    }
    _xi_sum = FLAGS_T*FLAGS_xi;

    // Actually load all the data 
    load_words(FLAGS_word_features_file);
    load_documents(FLAGS_ncrp_datafile);

    // DCHECK(tree_is_consistent());
    _ll = compute_log_likelihood();

    // NOTE: we need to do this in order to get the filename right...
    LOG(INFO) << "Initial state: " << current_state();

    VLOG(1) << "writing dictionary";
    write_dictionary();
    LOG(INFO) << "done initialize";
}

string ClusteredLDA::current_state() {
    _filename = _datafile_moniker;

    _filename += StringPrintf("-%s-T%d-K%d-alpha%f-eta%f-xi%f-seed%d", _word_features_moniker.c_str(), FLAGS_T, FLAGS_K,
            _alpha_sum / (double)_alpha.size(),
            _eta_sum / (double)_eta.size(),
            _xi_sum / (double)_xi.size(),
            FLAGS_random_seed);
    return StringPrintf("ll = %f (%f at %d) alpha = %f eta = %f xi = %f T = %d K = %d",
            _ll, _best_ll, _best_iter, 
            _alpha_sum / (double)_alpha.size(),
            _eta_sum / (double)_eta.size(),
            _xi_sum / (double)_xi.size(), FLAGS_T, FLAGS_K);
}

void ClusteredLDA::load_words(const string& input_file_name) {
    LOG(INFO) << "loading words from " << input_file_name;
    _unique_feature_count = 0;
    _total_feature_count = 0;
    _unique_word_count = 0;
    _total_word_count = 0;

    _unique_type_count = 0;

    _feature_name_to_id.set_empty_key(kEmptyStringKey);
    _feature_id_to_name.set_empty_key(kEmptyUnsignedKey);
    _word_name_to_id.set_empty_key(kEmptyStringKey);
    _word_id_to_name.set_empty_key(kEmptyUnsignedKey);
    _features.set_empty_key(kEmptyUnsignedKey);
    _cluster.set_empty_key(kEmptyUnsignedKey);

    _V.clear();
    _word_id_to_name.clear();

    // Keep track of the number of times we find each word
    vector<unsigned> found_count ;

    ifstream input_file(input_file_name.c_str());
    CHECK(input_file.is_open());

    string curr_line;
    while (true) {
        getline(input_file, curr_line);

        if (input_file.eof()) {
            break;
        }

        vector<string> features;
        curr_line = StringReplace(curr_line, "\n", "", true);

        SplitStringUsing(curr_line, "\t", &features);

        CHECK_GT(features.size(), 2) << "corrupt feature [" << curr_line << "]";

        // Read in the word name
        CHECK(_word_name_to_id.find(features[0]) == _word_name_to_id.end())
            << "found a duplicate word.";
        VLOG(1) << "found new word [" << features[0] << "]";
        _word_name_to_id[features[0]] = _unique_word_count;
        _word_id_to_name[_unique_word_count] = features[0];
        _unique_word_count += 1;

        // Read in the word type
        if (_type_name_to_id.find(features[1]) == _type_name_to_id.end()) {
            VLOG(1) << "found new type [" << features[1] << "]";
            _type_name_to_id[features[1]] = _unique_type_count;
            _type_id_to_name[_unique_type_count] = features[1];
            _unique_type_count += 1;
        }

        unsigned current_word_id = _word_name_to_id[features[0]];

        // Add in K clusters per word (this block is for type-dependent)
        if (FLAGS_type_dependent_clustering) {
            _word_id_to_type_id[current_word_id] = _type_name_to_id[features[1]];
            for (int k = 0; k < FLAGS_K; k++) {
                unsigned current_new_cluster = current_word_id*FLAGS_K+k;
                _master_cluster.push_back(CRP(0,0));
                _master_cluster[current_new_cluster].label = features[0];
                //_cluster[current_word_id].set_empty_key(kEmptyUnsignedKey);
                _cluster[_word_id_to_type_id[current_word_id]][current_new_cluster] = &_master_cluster[current_new_cluster];
            }
        } else {  // Add each cluster as a possibility for each word
            _word_id_to_type_id[current_word_id] = 0;  // Each word is type 0
            LOG(INFO) << "ignoring type info [" << features[1] << "]";
        }

        // Read in all this word's features
        vector<Feature> encoded_features;
        for (int k = 2; k < features.size(); k++) {
            vector<string> feature_tokens;

            SplitStringUsing(features.at(k), ":", &feature_tokens);
            // CHECK_EQ(feature_tokens.size(), 2);
            string feature;
            int freq;
            if (feature_tokens.size() == 2) {
                feature = feature_tokens.at(0);
                freq = atoi(feature_tokens.at(1).c_str());
            } else {
                // LOG(INFO) << "weird token [" << features.at(k) << "]";
                feature = "";
                for (int j = 0; j < feature_tokens.size() - 1; j++) {
                    feature += feature_tokens.at(j);
                }
                freq = atoi(feature_tokens.at(feature_tokens.size()-1).c_str());
            }

            // Handle new unique features
            if (_feature_name_to_id.find(feature) == _feature_name_to_id.end()) {
                _feature_name_to_id[feature] = _unique_feature_count;
                _unique_feature_count += 1;

                _feature_id_to_name[_feature_name_to_id[feature]] = feature;
                _eta.push_back(0);
                found_count.push_back(0);
                VLOG(1) << "found new feature [" << feature << "]";
            }
            _total_feature_count += freq;

            _V[_feature_name_to_id[feature]] += freq;

            encoded_features.push_back(Feature(_feature_name_to_id[feature], freq));

            // Take care of computing a per-word eta smoother prior (exponential
            // in rank?)
            _eta[_feature_name_to_id[feature]] += freq;
            found_count[_feature_name_to_id[feature]] += 1;

            // TODO: cluster initialization
        }
        _features[_word_name_to_id[features[0]]] = encoded_features;
        _total_word_count += 1;
    }

    _lV = _V.size();


    // Initialize the per-word dirichlet parameters
    CHECK_EQ(_eta.size(), _lV);
    // Compute the average rank for each word
    _eta_sum = 0;
    for (int l = 0; l < _lV; l++) {
        _eta[l] /= (double)found_count.at(l);
        double avg = _eta.at(l);
        _eta[l] = FLAGS_eta*pow(FLAGS_eta_prior_exponential, _eta[l]);
        _eta_sum += _eta[l];
        VLOG(1) << "prior for [" << _feature_id_to_name[l] << "] = " << _eta[l] << " (" << avg << ")";
    }
    VLOG(1) << "eta_sum is " << _eta_sum << " vs " << FLAGS_eta*_lV << " with flat prior";
    VLOG(1) << "eta_avg " << _eta_sum / (double)_eta.size();
}

void ClusteredLDA::load_documents(const string& input_file_name) {
    LOG(INFO) << "loading documents from " << input_file_name;
    _lD = 0;

    _document_name.set_empty_key(kEmptyUnsignedKey); 
    _document_id.set_empty_key(kEmptyStringKey);

    _D.clear();

    ifstream input_file(input_file_name.c_str());
    CHECK(input_file.is_open());

    string curr_line;
    while (true) {
        getline(input_file, curr_line);

        if (input_file.eof()) {
            break;
        }

        vector<string> words;
        vector<unsigned> encoded_words;
        curr_line = StringReplace(curr_line, "\n", "", true);

        SplitStringUsing(curr_line, "\t", &words);

        CHECK_GT(words.size(), 1) << "corrupt document [" << curr_line << "]";

        // Parse the document name first
        words[0] = StringReplace(words.at(0), "rpl_", "", false);
        words[0] = StringReplace(words.at(0), "RPL_", "", false);
        _document_name[_lD] = words[0];
        LOG(INFO) << "found new document [" << words[0] << "] " << _lD;
        _document_id[words[0]] = _lD;

        _D.push_back(NestedDocument(words[0], _lD));

        // Read in all the words and their associated features
        for (int i = 1; i < words.size(); i++) {
            CHECK_STRNE(words[i].c_str(), "");
            vector<string> word_tokens;

            SplitStringUsing(words.at(i), ":", &word_tokens);
            CHECK_EQ(word_tokens.size(), 2);
            string word = word_tokens.at(0);
            int freq = atoi(word_tokens.at(1).c_str());

            NestedDocument& last_doc = _D.back();

            if (_word_name_to_id.find(word) == _word_name_to_id.end()) {
                LOG(INFO) << "missing [" << word << "] in id table";
                continue;
            }

            unsigned current_word_id = _word_name_to_id[word];

            DCHECK(_features.find(current_word_id) != _features.end()) 
               << "missing word [" << word << "] in features table";

            // Add the word and its features and assign them randomly to
            // clusters
            unsigned d = last_doc._doc_id;
            VLOG(1) << "Adding " << word << " " << freq << " times with " << _features[current_word_id].size() << " features";
            for (int t = 0; t < freq; t++) {
                last_doc._words.push_back(NestedDocument::WordFeatures(current_word_id, word, _word_id_to_type_id[current_word_id]));
                last_doc._words.back().uniform_initialization();

                NestedDocument::WordFeatures& last_word = _D.back()._words.back();
                unsigned z = last_word._topic_indicator;
                unsigned w = last_word._cluster_indicator;
                unsigned w_uid = last_word._word_id; // the key of this word in the features table (not the cluster indicator)
                unsigned w_type_id = last_word._word_type_id; // the key of this word in the types table (not the cluster indicator)
                // unsigned w = _D.back()._words.back()._word_id;

                CHECK_GE(_topic[z].nw[w], 0);

                // Initialize the word->topic assignments
                _topic[z].nw[w] += 1;  // number of words in topic z equal to w
                _topic[z].nd[d] += 1;  // number of words in doc d with topic z
                _topic[z].nwsum += 1;  // number of words in topic z
                _nd[d] += 1;

                // Initialize the feature->cluster assignments
                for (int k = 0; k < _features[current_word_id].size(); k++) {
                    unsigned f = _features[current_word_id][k]._feature_id;
                    unsigned c = _features[current_word_id][k]._count;
                    _cluster[w_type_id][w]->nw[f] += c;
                    _cluster[w_type_id][w]->nwsum += c;
                }
            }


            for (int z = 0; z < FLAGS_T; z++) {
                _topic[z].ndsum += 1;
            }

            // if (d > 0) {
            //   resample_posterior_z_for(d);
            // } 
        }
        _lD += 1;
    }

    LOG(INFO) << "Loaded " << _lD << " documents with "
        << _total_word_count << " words (" << _unique_word_count << " unique) comprised of "
        << _total_feature_count << " features (" << _unique_feature_count << " unique) from "
        << input_file_name;
}


// Compute word->topic assignments (z) conditioned on the identities for each of
// the words (w)
void ClusteredLDA::resample_posterior_z_for(unsigned d, bool remove) {
    CHECK(false) << "remove";
    for (int n = 0; n < _D[d]._words.size(); n++) {  // loop over every word
        unsigned w = _D[d]._words[n]._cluster_indicator; // w is the latent word id
        unsigned z = _D[d]._words[n]._topic_indicator;

        // Resample the word assignment 

        // Compute the new level assignment #
        // #################################### Remove this word from the counts
        _topic[z].nw[w] -= 1;  // number of words in topic z equal to w
        _topic[z].nd[d] -= 1;  // number of words in doc d with topic z
        _topic[z].nwsum -= 1;  // number of words in topic z
        _nd[d] -= 1;  // number of words in doc d

        CHECK_GE(_topic[z].nwsum, 0);
        CHECK_GE(_topic[z].nw[w], 0);
        CHECK_GE(_nd[d], 0);
        CHECK_GT(_topic[z].ndsum, 0);

        vector<double> lposterior_z_dn;
        for (int l = 0; l < _topic.size(); l++) {
            lposterior_z_dn.push_back(log(_xi.at(w) + _topic[l].nw[w]) -
                    log(_xi_sum + _topic[l].nwsum) +
                    log(_alpha.at(l) + _topic[l].nd[d]) -
                    log(_alpha_sum + _nd[d]));
        }
        // Update the assignment
        _D[d]._words[n]._topic_indicator = sample_unnormalized_log_multinomial(&lposterior_z_dn);
        // VLOG(1) << "orig " << z << " new " << _D[d]._words[n]._topic_indicator;

        CHECK_LE(_D[d]._words[n]._topic_indicator, _topic.size());

        // Update the counts

        // Check to see that the default dictionary insertion works like we
        // expect
        // DCHECK(_topic[z].nw.find(w) != _topic[z].nw.end() || _topic[z].nw[w] == 0);
        // DCHECK(_topic[z].nd.find(d) != _topic[z].nd.end() || _topic[z].nd[d] == 0);
        
        z = _D[d]._words[n]._topic_indicator;

        _topic[z].nw[w] += 1;  // number of words in topic z equal to w
        _topic[z].nd[d] += 1;  // number of words in doc d with topic z
        _topic[z].nwsum += 1;  // number of words in topic z
        _nd[d]              += 1;  // number of words in doc d

        CHECK_GT(_topic[z].ndsum, 0);
    }
}

// Compute w conditional on z
void ClusteredLDA::resample_posterior_w_for(unsigned d) {
    for (int n = 0; n < _D[d]._words.size(); n++) {  // loop over every word
        unsigned w = _D[d]._words[n]._cluster_indicator;
        unsigned z = _D[d]._words[n]._topic_indicator;
        unsigned w_uid = _D[d]._words[n]._word_id; // the key of this word in the features table (not the cluster indicator)
        unsigned w_type_id = _D[d]._words[n]._word_type_id; // the key of this word in the types table (not the cluster indicator)

        // Resample the feature->cluster assignment 
        // First remove all this word's features from the correct cluster
        for (int k = 0; k < _features[w_uid].size(); k++) {
            unsigned f = _features[w_uid][k]._feature_id;
            unsigned c = _features[w_uid][k]._count;

            // Compute the new cluster assignment #
            // #################################### Remove this word from the counts
            _cluster[w_type_id][w]->nw[f] -= c;  // number of features in cluster z equal to f
            _cluster[w_type_id][w]->nwsum -= c;  // number of features in cluster z

            CHECK_GE(_cluster[w_type_id][w]->nwsum, 0);
            CHECK_GE(_cluster[w_type_id][w]->nw[w], 0);
        }
        _topic[z].nw[w] -= 1;  // number of words in topic z equal to w
        _topic[z].nwsum -= 1;  // number of words in cluster z

        CHECK_GE(_topic[z].nwsum, 0);
        CHECK_GE(_topic[z].nw[w], 0);

        // Do the cluster reassignment
        // TODO: XXX: likelihood is perhaps wrong here (topic part at least)
        vector<double> lposterior_w_dn;
        for (int l = 0; l < _cluster[w_type_id].size(); l++) {  // l is now the word
            double lp_w = 0;
            lp_w = log(_xi.at(l) + _topic[z].nw[l]) - log(_xi_sum + _topic[z].nwsum);
            for (int k = 0; k < _features[w_uid].size(); k++) {
                unsigned f = _features[w_uid][k]._feature_id;
                lp_w += log(_eta.at(f) + _cluster[w_type_id][l]->nw[f]);
            }
            lp_w -= _features[w_uid].size() * log(_eta_sum + _cluster[w_type_id][l]->nwsum); // normalize for features length
            lposterior_w_dn.push_back(lp_w);
        }
        // Update the assignment
        _D[d]._words[n]._cluster_indicator = sample_unnormalized_log_multinomial(&lposterior_w_dn);
        // VLOG(1) << "orig " << w << " new " << _D[d]._words[n]._cluster_indicator;

        CHECK_LE(_D[d]._words[n]._cluster_indicator, _cluster[w_type_id].size());

        // Finally add everything back in
        w = _D[d]._words[n]._cluster_indicator;
        for (int k = 0; k < _features[w_uid].size(); k++) {
            unsigned f = _features[w_uid][k]._feature_id;
            unsigned c = _features[w_uid][k]._count;
            // Update the counts

            // Check to see that the default dictionary insertion works like we
            // expect
            // DCHECK(_cluster[z].nw.find(w) != _cluster[z].nw.end() || _cluster[z].nw[w] == 0);
            // DCHECK(_cluster[z].nd.find(d) != _cluster[z].nd.end() || _cluster[z].nd[d] == 0);

            _cluster[w_type_id][w]->nw[f] += c;  // number of words in cluster z equal to f
            _cluster[w_type_id][w]->nwsum += c;  // number of words in cluster z
        }
        _topic[z].nw[w] += 1;  // number of words in topic z equal to w
        _topic[z].nwsum += 1;  // number of words in cluster z
    }
}

void ClusteredLDA::resample_posterior() {
    CHECK_GT(_lV, 0);
    CHECK_GT(_lD, 0);
    CHECK_GT(FLAGS_T, 0);
    CHECK_GT(FLAGS_K, 0);

    // if (FLAGS_learn_eta) {
    //    resample_posterior_eta();
    // }

    // Interleaved version
    for (int d = 0; d < _D.size(); d++) {
        // LOG(INFO) <<  "  resampling document " <<  d;
        resample_posterior_z_for(d, true);
        resample_posterior_w_for(d);
    }
}

double ClusteredLDA::compute_log_likelihood() {
    // Compute the log likelihood for the tree
    double log_lik = 0;
    LOG(INFO) << "todo validate ll";

    // Compute the log likelihood of the level assignments (correctly?)
    for (int d = 0; d < _D.size(); d++) {

        // LOG(INFO) << d << " " << _nd[d] << " " << _eta_sum;
        for (int n = 0; n < _D[d]._words.size(); n++) {
            unsigned w = _D[d]._words[n]._cluster_indicator;
            unsigned z = _D[d]._words[n]._topic_indicator;
            unsigned w_uid = _D[d]._words[n]._word_id; // the key of this word in the features table (not the cluster indicator)
            unsigned w_type_id = _D[d]._words[n]._word_type_id; // the key of this word in the types table (not the cluster indicator)

            // likelihood of drawing this word given the topics
            CHECK_LE(_topic[z].nw[w], _topic[z].nwsum);

            log_lik += log(_topic[z].nw[w]+_xi[w]) - log(_topic[z].nwsum+_xi_sum);
            CHECK_LE(log_lik, 0) << "log likelihood went positive for [" << _document_name[d] << "]";

            log_lik += log(_topic[z].nd[d]+_alpha[z]) - log(_nd[d]+_alpha_sum);
            CHECK_LE(log_lik, 0);

            // Likelihood of the features given the word
            for (int k = 0; k < _features[w_uid].size(); k++) {
                unsigned f = _features[w_uid][k]._feature_id;
                log_lik += log(_cluster[w_type_id][w]->nw[f] + _eta[f]) - log(_cluster[w_type_id][w]->nwsum + _eta_sum);
            }

        }
    }
    return log_lik;
}

// Write out a static dictionary required for decoding Gibbs samples
void ClusteredLDA::write_dictionary() {
    string filename = StringPrintf("%s.dictionary", get_base_name(_filename).c_str());

    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);


    // First write out the word code
    for (WordCode::const_iterator itr = _word_id_to_name.begin(); itr != _word_id_to_name.end(); itr++) {
        f << itr->first << "\t" << itr->second << endl;
    }
    f << "XXXXXXXX XXXXXXXX" << endl;
    // Now write out the feature code
    for (WordCode::const_iterator itr = _feature_id_to_name.begin(); itr != _feature_id_to_name.end(); itr++) {
        f << itr->first << "\t" << itr->second << endl;
    }
}



// Write out all the data in an intermediate format
void ClusteredLDA::write_data(string prefix) {
    string filename = StringPrintf("%s-%s.hlda", get_base_name(_filename).c_str(), prefix.c_str());

    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    f << current_state() << endl;

    // Write out the state in topic-major format.
    // First the topics
    for (int t = 0; t < _topic.size(); t++) {
        CRP& current = _topic[t];
        // Write out the node contents
        f << "topic " << t << "\t||\t" << current.ndsum << "\t||";

        for (WordToCountMap::iterator nw_itr = current.nw.begin();
                nw_itr != current.nw.end(); nw_itr++) {
            if (nw_itr->second > 0) {  // sparsify
                f << "\t" << nw_itr->first << ":" << nw_itr->second;
            }
        }
        f << "\t||";
        for (DocToWordCountMap::iterator nd_itr = current.nd.begin();
                nd_itr != current.nd.end(); nd_itr++) {
            if (nd_itr->second > 0) {  // sparsify
                f << "\t" << nd_itr->first << ":" << nd_itr->second;
            }
        }
        f << endl;
        // end writing out node contents
    }
    // Next the clusters
    for (int t = 0; t < _master_cluster.size(); t++) {
        CRP& current = _master_cluster[t];
        // Write out the node contents
        f << "cluster " << t << "\t||\t" << current.label << "\t||";

        for (WordToCountMap::iterator nw_itr = current.nw.begin();
                nw_itr != current.nw.end(); nw_itr++) {
            if (nw_itr->second > 0) {  // sparsify
                f << "\t" << nw_itr->first << ":" << nw_itr->second;
            }
        }
        f << "\t||";
        for (DocToWordCountMap::iterator nd_itr = current.nd.begin();
                nd_itr != current.nd.end(); nd_itr++) {
            if (nd_itr->second > 0) {  // sparsify
                f << "\t" << nd_itr->first << ":" << nd_itr->second;
            }
        }
        f << endl;
        // end writing out node contents
    }
}

