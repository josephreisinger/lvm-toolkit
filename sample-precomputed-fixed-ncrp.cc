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
// Like sample-fixed-ncrp but reads in the topic structure in a much more
// flexible way (assumes a file containing doc<tab>topic<tab>topic).
// then load topic assignments directly from the topic_assignments_file
// instead of inferring them from the hierarchy.

#include <math.h>
#include <time.h>

#include "ncrp-base.h"
#include "sample-fixed-ncrp.h"
#include "sample-precomputed-fixed-ncrp.h"

// Topic assignments file, if we've precomputed the topic assignments
DEFINE_string(topic_assignments_file,
            "",
            "file holding document<tab>topic<tab>topic... for every document");

// Number of additional topics to use over the labeled ones
DEFINE_int32(additional_noise_topics,
            0,
            "number of additional topics to include in addition to the labeled ones");

// Should we cull topics that only have one document?
DEFINE_bool(cull_unique_topics,
            true,
            "should we cull topics that only have one document?");

void NCRPPrecomputedFixed::add_crp_node(const string& name) {
    if (_node_to_crp.find(name) == _node_to_crp.end()) {
        // havent made a CRP node for this yet
        _node_to_crp[name] = new CRP(0, 0);
        _node_to_crp[name]->label = name;
        CHECK_EQ(_node_to_crp[name]->prev.size(), 0);
        VLOG(1) << "creating node [" << name << "]";
        _unique_nodes += 1;
    }
}

void NCRPPrecomputedFixed::load_precomputed_tree_structure(const string& filename) {
    LOG(INFO) << "loading tree";
    _unique_nodes = 0;
    _node_to_crp.set_empty_key(kEmptyStringKey);

    CHECK(!FLAGS_ncrp_skip_root);

    CHECK_STRNE(filename.c_str(), "");

    // These all cause problems.
    CHECK(FLAGS_use_dag);
    CHECK(!FLAGS_separate_path_assignments);
    CHECK(!FLAGS_sense_selection);
    CHECK(!FLAGS_learn_eta);
    CHECK(!FLAGS_gem_sampler);

    ifstream input_file(filename.c_str(), ios_base::in | ios_base::binary);

    // First add in the noise topics
    for (int i = 0; i < FLAGS_additional_noise_topics; i++) {
        add_crp_node(StringPrintf("NOISE_%d", i));
    }

    // First pass over the entire tree structure and create CRP nodes for each
    // of the unique node names
    string curr_line;
    while (true) {
        getline(input_file, curr_line);

        if (input_file.eof()) {
            break;
        }
        vector<string> tokens;
        curr_line = StringReplace(curr_line, "\n", "", true);

        // LOG(INFO) << curr_line;
        SplitStringUsing(curr_line, "\t", &tokens);

        string doc_name = tokens[0];

        if (_document_id.find(doc_name) != _document_id.end()) {
            // Add new nodes for each unseen topic and then build the _c vector for
            // the document, if necessary (adds a node specific to the document
            // as well)
            for (int i = 0; i < tokens.size(); i++) {
                add_crp_node(tokens.at(i));
            }
            // Build the _c vector
            unsigned d = _document_id[doc_name];
            for (int i = 1; i < tokens.size(); i++) {
                _c[d].push_back(_node_to_crp[tokens[i]]);
            }
            // Add in the noise topics
            for (int i = 0; i < FLAGS_additional_noise_topics; i++) {
                _c[d].push_back(_node_to_crp[StringPrintf("NOISE_%d", i)]);
            }
            // Resize _L and the alpha vector as needed
            if (_c[d].size() > _L) {
                _L    = _c[d].size();
                _maxL = _c[d].size();
                // Find the deepest branch
                if (_L > _alpha.size()) {
                    // Take care of resizing the vector of alpha hyperparameters, even
                    // though we currently don't learn them
                    for (int l = _alpha.size(); l < _L; l++) {
                        _alpha.push_back(FLAGS_ncrp_alpha);
                    }
                    _alpha_sum = _L*FLAGS_ncrp_alpha;
                }
            }
            for (int l = 0; l < _c[d].size(); l++) {
                // TODO: what is level used for?
                //_c[d][l]->level = l;
                _c[d][l]->ndsum += 1;
            }
        } else {
            VLOG(1) << "document [" << doc_name << "] from topics file is not in raw data source";
        }
    }

    VLOG(1) << "init tree done...";
}

void NCRPPrecomputedFixed::allocate_document(unsigned d) {
    // Do some sanity checking
    // Attach the words to this path

    string doc_name = _document_name[d];
    // CHECK(_node_to_crp.find(doc_name) != _node_to_crp.end())
    //    << "missing document [" << doc_name << "] in topics file"; 
    _total += 1;
    if(_node_to_crp.find(doc_name) == _node_to_crp.end()) {
        VLOG(1) << "missing document [" << doc_name << "] in topics file"; 
        _missing += 1;
        return;
    } 
    // CHECK(_c[d].size() > 0);
    //
    _nd[d] = 0;

    // Cull out all the topics where m==1 (we're the only document
    // referencing this topic)
    vector <CRP*> new_topics;
    for (int l = 0; l < _c[d].size(); l++) {
        CHECK_GT(_c[d][l]->ndsum, 0);
        if (_c[d][l]->ndsum == 1 && _c[d][l]->label != doc_name && FLAGS_cull_unique_topics) {
            VLOG(1) << "culling topic [" << _c[d][l]->label << "] from document [" << doc_name << "]";
            new_topics[0]->label += " | " + _c[d][l]->label;
            VLOG(1) << "              [" << new_topics[0]->label << "]";
        } else {
            LOG(INFO) << "keeping topic [" << _c[d][l]->label << "] for document  [" << doc_name << "] size " << _c[d][l]->ndsum;
            new_topics.push_back(_c[d][l]);
        }
    }

    _c[d] = new_topics;

    if (_c[d].empty()) {
        LOG(INFO) << "removing document [" << doc_name << "] since no non-unique topics"; 
        _missing += 1;
        // CHECK_GT(_c[d].size(), 0) << "failed topic check.";
        return;
    }

    for (int n = 0; n < _D[d].size(); n++) {
        // CHECK_GT(_c[d].size(), 0) << "[" << _document_name[d] << "] has a zero length path";
        unsigned w = _D[d][n];

        // set a random topic assignment for this guy
        _z[d][n] = sample_integer(_c[d].size());

        // test the initialization of maps
        CHECK(_c[d][_z[d][n]]->nw.find(w) != _c[d][_z[d][n]]->nw.end()
                || _c[d][_z[d][n]]->nw[w] == 0);
        CHECK(_c[d][_z[d][n]]->nd.find(d) != _c[d][_z[d][n]]->nd.end()
                || _c[d][_z[d][n]]->nd[d] == 0);

        _c[d][_z[d][n]]->nw[w] += 1;  // number of words in topic z equal to w
        _c[d][_z[d][n]]->nd[d] += 1;  // number of words in doc d with topic z
        _c[d][_z[d][n]]->nwsum += 1;  // number of words in topic z
        _nd[d]      += 1;  // number of words in doc d

        _total_words += 1;
    }
    // Incrementally reconstruct the tree (each time we add a document,
    // update its tree assignment based only on the previously added
    // documents; this results in a "fuller" initial tree, instead of one
    // fat trunk (fat trunks cause problems for mixing)
    if (d > 0) {
        LOG(INFO) << "resample posterior for " << doc_name;
        resample_posterior_z_for(d, true);
    }
}

void NCRPPrecomputedFixed::batch_allocation() {
    LOG(INFO) << "Doing precomputed tree ncrp batch allocation...";

    _missing = 0;
    _total = 0;

    // Load the precomputed tree structure (after we've loaded the document;
    // before we allocate the document)
    load_precomputed_tree_structure(FLAGS_topic_assignments_file);

    // Allocate the document
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;
        allocate_document(d);
    }
    LOG(INFO) << "missing " << _missing << " of " << _total;

    // NOTE: we need to do this in order to get the filename right...
    LOG(INFO) << "Initial state: " << current_state();

    VLOG(1) << "writing dictionary";
    write_dictionary();
}


// Write out a static dictionary required for decoding Gibbs samples
void NCRPPrecomputedFixed::write_dictionary() {
    string filename = StringPrintf("%s-%s-noise%d-%d.dictionary", get_base_name(_filename).c_str(), get_base_name(FLAGS_topic_assignments_file).c_str(), FLAGS_additional_noise_topics, FLAGS_random_seed);
    LOG(INFO) << "writing dictionary to [" << filename << "]";

    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    // First write out all the topics for each document that was kept in the
    // sampler
    set<CRP*> visited_nodes;  // keep track of visited things for the DAG case
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        if (!_c[d].empty()) {
            f << _document_name[d] << "\t" << d;
            for (int l = 0; l < _c[d].size(); l++) {
                f << "\t" << _c[d][l];
            }
            f << endl;
            // Write out any unvisited topics
            for (int l = 0; l < _c[d].size(); l++) {
                if (visited_nodes.find(_c[d][l]) == visited_nodes.end()) {
                    visited_nodes.insert(_c[d][l]);

                    CRP* current = _c[d][l];

                    f << current << "\t" << current->label << endl;
                }
            }
        }
    }

    f << endl;
    // Now write out the term dictionary
    for (WordCode::const_iterator itr = _word_id_to_name.begin(); itr != _word_id_to_name.end(); itr++) {
        f << itr->first << "\t" << itr->second << endl;
    }
}


// Write out all the data in an intermediate format
void NCRPPrecomputedFixed::write_data(string prefix) {
    string filename = StringPrintf("%s-%s-noise%d-%d-%s.hlda", get_base_name(_filename).c_str(), 
            get_base_name(FLAGS_topic_assignments_file).c_str(),
            FLAGS_additional_noise_topics,
            FLAGS_random_seed,
            prefix.c_str());
    VLOG(1) << "writing data to [" << filename << "]";

    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    f << current_state() << endl;

    // Write out all the topic-term distributions
    set<CRP*> visited_nodes;  // keep track of visited things for the DAG case
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;
        if (!_c[d].empty()) {
            // Write out any unvisited topics
            for (int l = 0; l < _c[d].size(); l++) {
                if (visited_nodes.find(_c[d][l]) == visited_nodes.end()) {
                    visited_nodes.insert(_c[d][l]);

                    CRP* current = _c[d][l];

                    // Write out the node contents
                    f << current << "\t||\t" << current->ndsum << "\t||";

                    for (WordToCountMap::iterator nw_itr = current->nw.begin();
                            nw_itr != current->nw.end(); nw_itr++) {
                        if (nw_itr->second > 0) {  // sparsify
                            f << "\t" << nw_itr->first << ":" << nw_itr->second;
                        }
                    }
                    f << "\t||";
                    for (DocToWordCountMap::iterator nd_itr = current->nd.begin();
                            nd_itr != current->nd.end(); nd_itr++) {
                        if (nd_itr->second > 0) {  // sparsify
                            f << "\t" << nd_itr->first << ":" << nd_itr->second;
                        }
                    }
                    f << endl;
                    // end writing out node contents
                }
            }
        }
    }
}

