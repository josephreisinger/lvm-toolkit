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
// Samples from the Nested Chinese Restaurant Process using a fixed topic
// structure. This model is more expressive in that the topic structure is not
// constrained to be a tree, but rather a digraph.

#include <math.h>
#include <time.h>

#include "ncrp-base.h"
#include "sample-fixed-ncrp.h"

// The GEM(m, \pi) distribution hyperparameter m, controls the "proportion of
// general words relative to specific words"
DEFINE_double(gem_m,
              0.1,
              "m reflects the proportion of general words to specific words");

// The GEM(m, \pi) hyperparameter \pi: reflects how strictly we expect the
// documents to adhere to the m proportions.
DEFINE_double(gem_pi,
              0.1,
              "reflects our confidence in the setting m");

// The file path from which to load the topic structure. The file must be
// encoded as one connection per line, child <tab> parent.
DEFINE_string(tree_structure_file,
              "",
              "the file from which to load the fixed tree");

// Whether or not to use the GEM sampler. The Multinomial sampler currently is
// more flexible as it allows the tree structure to be a DAG; the GEM sampler
// might not work yet with DAGs.
DEFINE_bool(gem_sampler,
            false,
            "true = use gem sampler, false = use multinomial sampler");

// If unset, then just throw away extra edges that cause nodes to have multiple
// parents. Enforcing a tree topology.
DEFINE_bool(use_dag,
            false,
            "should we build a tree or use the full dag (multinomial or GEM)");

// Should non-WN class nodes have words assigned to them? If not, then all
// topics will start with wn_
DEFINE_bool(fold_non_wn,
            false,
            "should non-WN classes be folded into their WN parent senses?");

// Should we perform variable selection (i.e. attribute rejection) based on
// adding a single "REJECT" node with a uniform distribution over the
// vocabulary to each topic list?
DEFINE_bool(use_reject_option,
            false,
            "should we perform variable selection with a reject option");

// Should the hyperparameters on the vocabulary Dirichlet (eta) be learned. For
// now this uses moment matchin to perform the updates.
DEFINE_bool(learn_eta,
            false,
            "should we adapt the etas using moment-matching");

// Should all the path combinations to the root be separated out into different
// documents? DAG only.
DEFINE_bool(separate_path_assignments,
            false,
            "make the dag execution more like the tree");

// Should we try to learn a single best sense from a list of senses?
DEFINE_bool(sense_selection,
            false,
            "perform sense selection; needs a hierarchy with multiple attachments");

GEMNCRPFixed::GEMNCRPFixed(double m,
        double pi)
: _gem_m(m), _pi(pi), _maxL(0) {
    _L = 3;  // HACK: for now we need an initial depth (for the first data pt)
    _maxL = _L;

    CHECK_GE(_gem_m, 0.0);
    CHECK_LE(_gem_m, 1.0);
}


string GEMNCRPFixed::current_state() {
    // HACK: put this in here for now since it needs to get updated whenever
    // maxL changes
    _filename = FLAGS_ncrp_datafile;

    if (FLAGS_use_dag) {
        _filename += "-DAG";
    }

    if (FLAGS_gem_sampler) {
        _filename += StringPrintf("-L%d-m%f-pi%f-eta%f-zpi%d",
                _maxL, _gem_m, _pi,
                _eta_sum / (double)_eta.size(),
                FLAGS_ncrp_z_per_iteration);
        return StringPrintf("ll = %f (%f at %d) (%d nodes) m = %f pi = %f eta = %f L = %d",
                _ll, _best_ll, _best_iter, _unique_nodes, _gem_m, _pi,
                _eta_sum / (double)_eta.size(), _maxL);
    } else {
        _filename += StringPrintf("-L%d-alpha%f-eta%f-zpi%d", _maxL,
                _alpha_sum / (double)_alpha.size(),
                _eta_sum / (double)_eta.size(),
                FLAGS_ncrp_z_per_iteration);
        return StringPrintf("ll = %f (%f at %d) (%d nodes) alpha = %f eta = %f L = %d",
                _ll, _best_ll, _best_iter, _unique_nodes,
                _alpha_sum / (double)_alpha.size(),
                _eta_sum / (double)_eta.size(), _maxL);
    }
}

void GEMNCRPFixed::load_precomputed_tree_structure(const string& filename) {
    google::dense_hash_map<string, CRP*> node_to_crp;
    node_to_crp.set_empty_key(kEmptyStringKey);

    CHECK(!FLAGS_ncrp_skip_root);

    CHECK_STRNE(filename.c_str(), "");

    // These all cause problems.
    CHECK(FLAGS_use_dag);
    CHECK(!FLAGS_separate_path_assignments);
    CHECK(!FLAGS_sense_selection);
    CHECK(!FLAGS_learn_eta);
    CHECK(!FLAGS_gem_sampler);


    // Actually read in the file
    ifstream input_file(filename.c_str());
    CHECK(input_file.is_open());

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
            // the document, if necessary
            for (int i = 0; i < tokens.size(); i++) {
                if (node_to_crp.find(tokens[i]) == node_to_crp.end()) {
                    // havent made a CRP node for this yet
                    node_to_crp[tokens[i]] = new CRP(0, 0);
                    node_to_crp[tokens[i]]->label = tokens[i];
                    CHECK_EQ(node_to_crp[tokens[i]]->prev.size(), 0);
                    LOG(INFO) << "creating node [" << tokens[i] << "]";
                }
            }
            // Build the _c vector
            unsigned d = _document_id[doc_name];
            for (int i = 1; i < tokens.size(); i++) {
                _c[d].push_back(node_to_crp[tokens[i]]);
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

    // Do some sanity checking
    // Attach the words to this path
    unsigned total = 0;
    unsigned missing = 0;
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        string doc_name = _document_name[d];
        // CHECK(node_to_crp.find(doc_name) != node_to_crp.end())
        //    << "missing document [" << doc_name << "] in topics file"; 
        total += 1;
        if(node_to_crp.find(doc_name) == node_to_crp.end()) {
            LOG(INFO) << "missing document [" << doc_name << "] in topics file"; 
            missing += 1;
            continue;
        } 
        // CHECK(_c[d].size() > 0);
        //
        _nd[d] = 0;

        // Cull out all the topics where m==1 (we're the only document
        // referencing this topic)
        vector <CRP*> new_topics;
        for (int l = 0; l < _c[d].size(); l++) {
            CHECK_GT(_c[d][l]->ndsum, 0);
            if (_c[d][l]->ndsum == 1) {
                LOG(INFO) << "culling topic [" << _c[d][l]->label << "] from document [" << doc_name << "]";
            } else {
                LOG(INFO) << "keeping topic [" << _c[d][l]->label << "] for document  [" << doc_name << "] size " << _c[d][l]->ndsum;
                new_topics.push_back(_c[d][l]);
            }
        }
        _c[d] = new_topics;

        if (_c[d].empty()) {
            LOG(INFO) << "removing document [" << doc_name << "] since no non-unique topics"; 
            missing += 1;
            // CHECK_GT(_c[d].size(), 0) << "failed topic check.";
            continue;
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
            LOG(INFO) << "done";
        }
    }
    LOG(INFO) << "missing " << missing << " of " << total;


    VLOG(1) << "init done...";
}


void GEMNCRPFixed::load_tree_structure(const string& filename) {
    google::dense_hash_map<string, CRP*> node_to_crp;
    node_to_crp.set_empty_key(kEmptyStringKey);

    CHECK_STRNE(FLAGS_ncrp_datafile.c_str(), "");

    // Check for some incompatible input parameter settings
    CHECK(!(FLAGS_separate_path_assignments && FLAGS_sense_selection));
    CHECK(!(!FLAGS_use_dag && FLAGS_sense_selection));

    ifstream input_file(filename.c_str());
    CHECK(input_file.is_open());

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
        //CHECK_EQ(x, 0);

        // LOG(INFO) << curr_line;
        SplitStringUsing(curr_line, "\t", &tokens);

        CHECK_GE(tokens.size(), 2);
        CHECK_LE(tokens.size(), 3);

        string source = tokens[0];
        LowerString(&source);
        source = StringReplace(source, "rpl_", "", false);

        string dest   = tokens[1];
        LowerString(&dest);
        dest = StringReplace(dest, "rpl_", "", false);


        if (node_to_crp.find(source) == node_to_crp.end()) {
            // havent made a CRP node for this yet
            node_to_crp[source] = new CRP(0, 0);
            node_to_crp[source]->label = source;
            CHECK_EQ(node_to_crp[source]->prev.size(), 0);
        }
        if (node_to_crp.find(dest) == node_to_crp.end()) {
            // havent made a CRP node for this yet
            node_to_crp[dest] = new CRP(0, 0);
            node_to_crp[dest]->label = dest;
            CHECK_EQ(node_to_crp[dest]->prev.size(), 0);
        }

        // Set up the pointers correctly so that the tree/dag gets built
        if (node_to_crp[source]->prev.size() > 0 && !FLAGS_use_dag) {
            // overwrite the existing connection
            LOG(INFO) << "overwriting [" << source << "] -> ["
                << node_to_crp[source]->prev[0]->label << "]";
            // Delete ourselves from our parent
            node_to_crp[source]->remove_from_parents();
            node_to_crp[source]->prev[0] = node_to_crp[dest];
        } else {
            node_to_crp[source]->prev.push_back(node_to_crp[dest]);
        }

        // Make sure this edge isn't already there
        CHECK(find(node_to_crp[dest]->tables.begin(),
                    node_to_crp[dest]->tables.end(),
                    node_to_crp[source]) == node_to_crp[dest]->tables.end());

        node_to_crp[dest]->tables.push_back(node_to_crp[source]);
        VLOG(1) << "connecting [" << source << "] -> [" << dest << "]";

        // Set up the node frequency info if necessary
        if (tokens.size() == 3) {
            CHECK(FLAGS_sense_selection); // Only sense selection can handle extra
            // input
            // Make sure that this is really a flat class
            CHECK(_document_id.find(source) != _document_id.end());

            double freq = strtod(tokens[2].c_str(), NULL);
            _log_node_freq[_document_id[source]][node_to_crp[dest]] = log(freq);
        }
    }

    // NOTE: this won't work as well in the DAG case
    // HACK to find the root
    VLOG(1) << "finding root...";
    _ncrp_root = node_to_crp.begin()->second;
    while (_ncrp_root->prev.size() > 0) {
        CHECK(_ncrp_root);
        CHECK(_ncrp_root->prev[0]);
        _ncrp_root = _ncrp_root->prev[0];
        LOG(INFO) << _ncrp_root->label;
    }

    // Now remove "excess" nodes in the form of chains and trim off nodes with
    // no non-wn children
    // NOTE: this has to come before the _c assignments are made
    contract_tree();

    if (FLAGS_use_reject_option) {
        _reject_node = new CRP(0,0); // add a REJECT topic
        _reject_node->label = "REJECT";
    }

    _total_words = 0;
    // Compute the _c assigments for each document and attach the words
    for (google::dense_hash_map<string, CRP*>::iterator itr = node_to_crp.begin();
            itr != node_to_crp.end();
            itr++) {
        string node_name = itr->first;
        CRP*   node      = itr->second;

        // Split the node name into its constituent classes and construct the
        // path assignment vector for each.
        vector<string> name_tokens;
        SplitStringUsing(node_name, " ", &name_tokens);

        // Loop over all the classes/concepts packaged into this node
        for (int i = 0; i < name_tokens.size(); i++) {
            // A node label may contain more than one class (using my clusterer)
            if (_document_id.find(name_tokens[i]) != _document_id.end()) {
                // add this document if its not an intermediate node
                VLOG(1) << "populating [" << name_tokens[i] << "]";
                unsigned d = _document_id[name_tokens[i]];

                CHECK_EQ(_c[d].size(), 0);

                // Perform a small graph structure sanity check: the current node must
                // exist in the next pointers of each of its parents
                for (int k = 0; k < node->prev.size(); k++) {
                    vector<CRP*>::iterator p = find(node->prev[k]->tables.begin(),
                            node->prev[k]->tables.end(), node);
                    // we must have existed in the previous
                    CHECK(p != node->prev[k]->tables.end())
                        << "Couldn't find [" << node->label << " in prev[" << k
                        << "] = [" << node->prev[k]->label;
                }

                // Collecting the set of nodes on this topic path is more
                // complex in the DAG case, we must traverse up each parent
                // branch and then merge the resulting set of nodes
                if (FLAGS_use_dag) {
                    if (FLAGS_separate_path_assignments) {
                        // In this case we split all the combinations of path segments to
                        // the root into individual paths and add new documents to
                        // compensate.
                        vector< vector<CRP*> > paths;
                        build_separate_path_assignments(node, &paths);

                        CHECK_GT(paths.size(), 0);

                        _c[d] = paths.at(0);

                        CHECK(false) << "this stuff probably won't work because _lD isn't back() anymore";
                        for (int i = 1; i < paths.size(); i++) {
                            _document_name[_lD] = _document_name[d];

                            // Create a new document for this path
                            _D[_lD] = _D[d];
                            // create a new _c entry
                            _c[_lD] = paths.at(i);
                            // Increment the total doc size
                            _lD += 1;
                        }
                    } else {
                        // generate the path assignment resulting from attaching
                        // node to its parent
                        if (FLAGS_sense_selection) {
                            build_path_assignments(node, &_c[d], 0);

                            // TODO: flesh out this and initialize c, z shadow
                            for (int s = 1; s < node->prev.size(); s++) {
                                build_path_assignments(node, &_c_shadow[d][s-1], s);
                            }
                        } else {
                            build_path_assignments(node, &_c[d], -1);
                        }
                    }
                } else {
                    // using 0 forces graft not to create new nodes
                    graft_path_at(node, &_c[d], 0);
                    // LOG(INFO) << node->label;
                    // for (int  k = 0; k < _c[d].size(); k++) {
                    //   LOG(INFO) << "  " << _c[d][k]->label;
                    // }

                    CHECK_EQ(_c[d].back(), node);
                }

                // make sure a node was allocated
                CHECK_GT(_c[d].size(), 0);

                // Add in a "REJECT" node if necessary
                if (FLAGS_use_reject_option) {
                    VLOG(1) << "Adding a REJECT node";
                    _c[d].push_back(_reject_node);
                }
            } else {
                VLOG(1) << "not populating [" << name_tokens[i] << "]";
            }
        }
    }
    VLOG(1) << "making path assignments...";
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        // Now compute the levels and document counts
        // TODO: the ->level variable is wrong for the DAG, but this
        // should not be a problem as long as we don't resample the path
        // assignments
        CHECK(_c[d].size() > 0) << "path [" << _document_name[d] << "] is size 0";
        for (int l = 0; l < _c[d].size(); l++) {
            _c[d][l]->level = l;
            _c[d][l]->ndsum += 1;
        }
        if (_c[d].size() > _L) {
            _L    = _c[d].size();
            _maxL = _c[d].size();
        }
        // Set the m and level values for the CRP nodes along the shadow paths as
        // well
        if (FLAGS_sense_selection) {
            for (int s = 0; s < node_to_crp[_document_name[d]]->prev.size()-1; s++) {
                for (int l = 0; l < _c_shadow[d][s].size(); l++) {
                    _c_shadow[d][s][l]->level = l;
                    _c_shadow[d][s][l]->ndsum += 1;
                }
                if (_c_shadow[d][s].size() > _L) {
                    _L    = _c_shadow[d][s].size();
                    _maxL = _c_shadow[d][s].size();
                }
            }
        }

        // Find the deepest branch
        if (_L > _alpha.size()) {
            // Take care of resizing the vector of alpha hyperparameters, even
            // though we currently don't learn them
            for (int l = _alpha.size(); l < _L; l++) {
                _alpha.push_back(FLAGS_ncrp_alpha);
            }
            _alpha_sum = _L*FLAGS_ncrp_alpha;
        }

        // Attach the words to this path
        _nd[d] = 0;

        for (int n = 0; n < _D[d].size(); n++) {
            CHECK_GT(_c[d].size(), 0) << "[" << _document_name[d] << "] has a zero length path";
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


            // Don't attach words to the shadow paths, but do set up their initial
            // level assignments.
            if (FLAGS_sense_selection) {
                for (int s = 0; s < node_to_crp[_document_name[d]]->prev.size()-1; s++) {
                    CHECK_GT(_c_shadow[d][s].size(), 0) << "[" << _document_name[d] << "] has a zero length path";
                    // set a random topic assignment for this guy
                    _z_shadow[d][s][n] = sample_integer(_c_shadow[d][s].size());
                    // Don't actually add the words in, since this is a shadow
                }
            }
        }

        // TODO: the below doesn't work if the tree has multiple
        // roots or if its a DAG
        if (_c[d].back() != _c[0].back()) {
            LOG(WARNING) << "found a tree rooted at [" << _c[d].back()->label
                << "] instead of [" << _c[0].back()->label << "]";
            // if (!FLAGS_use_dag) {
            //   CHECK_EQ(_c[d][0], _c[0][0]);  // force abort if tree-based
            // }
        }

        // Incrementally reconstruct the tree (each time we add a document,
        // update its tree assignment based only on the previously added
        // documents; this results in a "fuller" initial tree, instead of one
        // fat trunk (fat trunks cause problems for mixing)
        if (d > 0) {
            resample_posterior_z_for(d, true);
        }
    }
    CHECK_EQ(_ncrp_root->prev.size(), 0);
    VLOG(1) << "init done...";
}

// This should only be called on nodes inserted into WN. It removes node from
// whatever parents that it had and then reattaches it to parent, yielding the
// new _c path assignment that results.
void GEMNCRPFixed::build_path_assignments(CRP* node, vector<CRP*>* c, int sense_index) {
    // First attach node to parent
    // CHECK(node->prev.size() == 1);  // attach to only one sense...
    // node->remove_from_parents();
    // node->prev[0] = parent;
    // parent->tables.push_back(node);
    VLOG(1) << "[" << sense_index << "] " << node->label << ":";

    // Perform a level-by-level traversal up the tree, building
    // the vector of assignments
    deque<CRP*> node_queue;
    node_queue.push_back(node);
    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();
        // First check to see if we should remove the non WN concept nodes from
        // the list of topics
        if (!FLAGS_fold_non_wn || current->label.find("wn_") == 0) {
            // If this node is not yet in d's set of parents
            if (find(c->begin(), c->end(), current) == c->end()) {
                c->insert(c->begin(), current);
                VLOG(1) << "  [" << current->label << "]";
            }
        }
        // Add all the prev paths if sense_index < 0
        if (current != node || sense_index < 0) {
            node_queue.insert(node_queue.end(), current->prev.begin(),
                    current->prev.end());
        } else {
            // Otherwise add only the indexed path
            node_queue.push_back(current->prev[sense_index]);
        }
    }
}

// Builds a vector of the path combinations from this node to the root of the
// DAG.
void GEMNCRPFixed::build_separate_path_assignments(CRP* node, vector< vector<CRP*> >* paths) {
    // First attach node to parent
    // CHECK(node->prev.size() == 1);  // attach to only one sense...
    // node->remove_from_parents();
    // node->prev[0] = parent;
    // parent->tables.push_back(node);
    //

    // Perform a level-by-level traversal up the tree, building
    // the vector of assignments
    vector<CRP*> c;
    c.push_back(node);
    paths->push_back(c);
    bool all_ended = false;
    while (!all_ended) {
        all_ended = true;
        for (int i = 0; i < paths->size(); i++) {
            CRP * current = paths->at(i).back();
            // If the last element in the ith path has parents
            if (!current->prev.empty()) {
                paths->at(i).push_back(current->prev[0]);
                for (int k = 1; k < current->prev.size(); k++) {
                    paths->push_back(paths->at(i));
                    paths->back().back() = current->prev[k];
                }
                all_ended = false;
            }
        }
    }

    // Remove non wn nodes if we have to fold
    for (int i = 0; i < paths->size(); i++) {
        VLOG(1) << "[" << i << "] " << node->label << ":";
        if (FLAGS_fold_non_wn) {
            for (vector<CRP*>::iterator itr = paths->at(i).begin();
                    itr != paths->at(i).end(); itr++) {
                if ((*itr)->label.find("wn_") != 0) {
                    // XXX: for now this is a hack that assumes paths can contain only a
                    // single non wn_ node
                    paths->at(i).erase(itr);
                    break;
                }
            }
        }
        for (vector<CRP*>::iterator itr = paths->at(i).begin();
                itr != paths->at(i).end(); itr++) {
            VLOG(1) << "  [" << (*itr)->label << "]";
        }
    }
}

// Perform a level-by-level tree contration. This simplifies things, but may
// remove some of the explanatory power of hLDA (e.g., two different
// children containing a shared path with a chain could put different mass
// on the elements of the chain, allowing for a finer degree of expression)
void GEMNCRPFixed::contract_tree() {
    VLOG(1) << "contracting tree...";
    // TODO: this iteratively contracts the tree, removing nodes with no non
    // wn_ children. Could obviously be optimized
    bool contracted = true;
    while (contracted) {
        contracted = false;

        deque<CRP*> node_queue;
        set<CRP*> visited;
        node_queue.push_back(_ncrp_root);
        while (!node_queue.empty()) {
            CRP* current = node_queue.front();
            node_queue.pop_front();

            CHECK(current);

            if (visited.find(current) != visited.end()) {
                continue;
            }
            visited.insert(current);

            // remove this node if it has no non-wn children
            //
            if (current->tables.size() == 0 &&
                    current->label.find("wn_") == 0) {
                VLOG(1) << "pruned [" << current->label << "]";
                // for (int i = 0; i < current->prev.size(); i++) {
                //   VLOG(1) << "  [" << current->prev[i]->label << "]";
                // for (int k = 0; k < current->prev[i]->tables.size(); k++) {
                //   VLOG(1) << "    [" << current->prev[i]->tables[k]->label << "]";
                // }
                // }
                current->remove_from_parents();  // this will remove from parent
                contracted = true;
            }
            // try to merge nodes if possible
            if (current->tables.size() == 1 && current->tables.at(0)->prev.size() == 1) {
                CRP* target = current->tables.at(0);
                // if we have exactly one child and one parent
                if (target->tables.size() > 0) {
                    // don't collapse into leaves (non-wn ones at least)
                    VLOG(1) << "merged [" << current->label << "] with ["
                        << target->label << "]";

                    CHECK_EQ(target->prev.at(0), current);

                    target->label = current->label + "\\n" + target->label;
                    target->prev = current->prev;

                    // update the prev pointers
                    for (int i = 0; i < current->prev.size(); i++) {
                        // VLOG(1) << "  update children [" << current->tables[i]->label
                        //         << "]";
                        // First remove the old parent from each of the grandchildren's
                        // previous lists
                        safe_remove_crp(&current->prev[i]->tables, current);

                        // Now add the new previous pointer
                        current->prev[i]->tables.push_back(target);
                    }

                    // THis is the old code; it merges the child into current and removes
                    // the child -- this is bad for sense selection
                    /*current->label += "\\n" + current->tables[0]->label;
                      current->tables = current->tables.at(0)->tables;  // copy the tables

                    // update the prev pointers
                    for (int i = 0; i < current->tables.size(); i++) {
                    // VLOG(1) << "  update children [" << current->tables[i]->label
                    //         << "]";
                    // First remove the old parent from each of the grandchildren's
                    // previous lists
                    safe_remove_crp(&current->tables[i]->prev, old_parent);

                    // Now add the new previous pointer
                    current->tables[i]->prev.push_back(current);
                    }*/

                    // TODO: should we delete the node we contracted?

                    contracted = true;
                    break;
                }
            }
            node_queue.insert(node_queue.end(), current->tables.begin(),
                    current->tables.end());
        }
    }
}

// Performs a single document's level assignment resample step There is some
// significant subtly to this compared to the fixed depth version. First, we no
// longer have the guarantee that all documents are attached to leaves. Some
// documents may now stop at interior nodes of our current tree. This is ok,
// since the it just means that we haven't assigned any words to lower levels,
// and hence we haven't needed to actually assert what the path is. Anyway,
// since the level assignments can effectively change length here, we need to
// get more child nodes from the nCRP on the fly.
void GEMNCRPFixed::resample_posterior_z_for(unsigned d, vector<CRP*>& cd, WordToCountMap& zd) {
    // CHECK_EQ(_L, -1);  // HACK to make sure we're not using _L

    for (int n = 0; n < _D[d].size(); n++) {  // loop over every word
        unsigned w = _D[d][n];
        // Compute the new level assignment #
        // #################################### Remove this word from the counts
        cd[zd[n]]->nw[w] -= 1;  // number of words in topic z equal to w
        cd[zd[n]]->nd[d] -= 1;  // number of words in doc d with topic z
        cd[zd[n]]->nwsum -= 1;  // number of words in topic z
        _nd[d] -= 1;  // number of words in doc d

        CHECK_GE(cd[zd[n]]->nwsum, 0);
        CHECK_GE(cd[zd[n]]->nw[w], 0);
        CHECK_GE(_nd[d], 0);
        CHECK_GT(cd[zd[n]]->ndsum, 0);

        vector<double> lposterior_z_dn;
        if (FLAGS_gem_sampler) {
            CHECK(!FLAGS_use_reject_option)
                << "Reject option not implemented with gem_sampler";
            CHECK(false) << "GEM sampler might not work new eta,alpha changes FIXL2";
            // ndsum_above[k] is #[z_{d,-n} >= k]
            vector<unsigned> ndsum_above;
            ndsum_above.resize(cd.size());
            ndsum_above[cd.size()-1] = cd.back()->nd[d];
            // LOG(INFO) << "ndsum_above[" << cd.size()-1 << "] = "
            //           << ndsum_above.back();
            for (int l = cd.size()-2; l >= 0; l--) {
                // TODO: optimize this
                ndsum_above[l] = cd[l]->nd[d] + ndsum_above[l+1];
                // LOG(INFO) << "ndsum_above[" << l << "] = " << ndsum_above[l];
            }

            // Here we assign probabilities to all the "finite" options, e.g. all
            // the levels up to the current maximum level for this document. TODO:
            // this can be optimized quite extensively
            double V_j_sum = 0;
            unsigned total_nd = 0;
            for (int l = 0; l < cd.size(); l++) {
                // check that ["doesnt exist"]->0
                //DCHECK(cd[l]->nw.find(w) != cd[l]->nw.end() || cd[l]->nw[w] == 0);
                //DCHECK(cd[l]->nd.find(d) != cd[l]->nd.end() || cd[l]->nd[d] == 0);
                total_nd += cd[l]->nd[d];

                double lp_w_dn = log(_eta[w] + cd[l]->nw[w]) -
                    log(_eta_sum + cd[l]->nwsum);
                double lp_z_dn = log(_pi*(1-_gem_m) + cd[l]->nd[d]) -
                    log(_pi + ndsum_above[l]) + V_j_sum;

                lposterior_z_dn.push_back(lp_w_dn + lp_z_dn);
                // LOG(INFO) << l << " " << lp_w_dn << " + " << lp_z_dn
                //           << " = " << lposterior_z_dn[l];
                // LOG(INFO) << "  " << V_j_sum;

                if (l < cd.size()-1) {
                    V_j_sum += log(_gem_m*_pi + ndsum_above[l+1]) -
                        log(_pi + ndsum_above[l]);
                }
            }
            //DCHECK_EQ(total_nd, _nd[d]);
        } else {  // multinomial sampler
            double alpha_sum_c_d = 0;
            for (int l = 0; l < cd.size(); l++) {
                //LOG(INFO) << "jj" << n << l;
                alpha_sum_c_d += _alpha.at(l);
            }
            for (int l = 0; l < cd.size(); l++) {
                if (FLAGS_use_reject_option && cd[l]->label.find("REJECT") == 0) {
                    // This version puts a uniform distribution over the vocabulary in
                    // the reject node, but allows each document to choose its "affinity"
                    // for putting things there

                    // XXX: THis is the original version with _L wrong
                    // lposterior_z_dn.push_back(-log(_lV) +
                    //                           log(_alpha + cd[l]->nd[d]) -
                    //                           log(_L*_alpha + _nd[d]));


                    // This version is uniform over the vocab
                    lposterior_z_dn.push_back(-log(_lV) +
                            log(_alpha.at(l) + cd[l]->nd[d]) -
                            log(alpha_sum_c_d + _nd[d]));

                    // Uniform over the document's vocab
                    // lposterior_z_dn.push_back(-log(_D[d].size()) +
                    //                           log(norm_alpha + cd[l]->nd[d]) -
                    //                           log(_alpha + _nd[d]));
                } else {
                    // XXX: THis is the original version with _L wrong
                    // lposterior_z_dn.push_back(log(_eta + cd[l]->nw[w]) -
                    //                           log(_eta_sum + cd[l]->nwsum) +
                    //                           log(_alpha + cd[l]->nd[d]) -
                    //                           log(_L*_alpha + _nd[d]));
                    lposterior_z_dn.push_back(log(_eta.at(w) + cd.at(l)->nw[w]) -
                            log(_eta_sum + cd.at(l)->nwsum) +
                            log(_alpha.at(l) + cd[l]->nd[d]) -
                            log(alpha_sum_c_d + _nd[d]));
                }
            }
        }

        // Update the assignment
        zd[n] = sample_unnormalized_log_multinomial(&lposterior_z_dn);

        CHECK_LE(zd[n], cd.size());

        // Update the counts

        // Check to see that the default dictionary insertion works like we
        // expect
        // DCHECK(cd[zd[n]]->nw.find(w) != cd[zd[n]]->nw.end() || cd[zd[n]]->nw[w] == 0);
        // DCHECK(cd[zd[n]]->nd.find(d) != cd[zd[n]]->nd.end() || cd[zd[n]]->nd[d] == 0);

        cd[zd[n]]->nw[w] += 1;  // number of words in topic z equal to w
        cd[zd[n]]->nd[d] += 1;  // number of words in doc d with topic z
        cd[zd[n]]->nwsum += 1;  // number of words in topic z
        _nd[d]              += 1;  // number of words in doc d

        CHECK_GT(cd[zd[n]]->ndsum, 0);
    }
}

// When doing sense selection, we need to choose between possible alternative
// document->WN attachments. The way to do this is to keep multiple copies of
// te same document with "shadow" level assignments. When we resample the sense
// attachment, we first calculate the probability of the non-shadow copy, then
// remove all the words and iterate over the shadows, adding in all the words
// at their levels, recalculating the shadow level assignments, and then
// calculating the probability. 
void GEMNCRPFixed::resample_posterior_c_for(unsigned d) {
    vector<double> lp_c_d;  // log-probability of this branch c_d

    CHECK(!FLAGS_separate_path_assignments)
        << "Can't use separate path assignments when learning sense";

    // First compute the probability for the original non-shadow document
    double lf = _log_node_freq[d][_c[d].back()];
    lp_c_d.push_back(compute_path_probability_for(d,_c[d])+lf);

    remove_all_words_from(d, _c[d], _z[d]);

    // Now compute the probability for each of the shadow docments
    for (int s = 0; s < _z_shadow[d].size(); s++) {
        add_all_words_from(d, _c_shadow[d][s], _z_shadow[d][s]);

        resample_posterior_z_for(d, _c_shadow[d][s], _z_shadow[d][s]); // update the z assignments to be fair

        double lf = _log_node_freq[d][_c_shadow[d][s].back()];
        double lp = compute_path_probability_for(d,_c_shadow[d][s]);
        lp_c_d.push_back(lp+lf);
        //LOG(INFO) << "lf for [" << _document_name[d] << "] attaching at [" << _c_shadow[d][s].back()->label << "] is " << lf << "  : " << lp << " = " << lp+lf;
        CHECK_LE(lf,0);  // make sure lf is a valid probability

        remove_all_words_from(d, _c_shadow[d][s], _z_shadow[d][s]);
    }

    // Actually do the sampling (select a new path)
    int index = sample_unnormalized_log_multinomial(&lp_c_d);

    // Declare a path assignment winner and redirect _z[d] and re add the counts
    if (index > 0) {
        //VLOG(1) << "swapping [" << _document_name[d] << "] from "
        //        << _c[d].back()->label << " to " << _c_shadow[d][index-1].back()->label;
        // Swap the old _z[d] with the shadow

        WordToCountMap temp = _z_shadow[d][index-1];
        vector<CRP*> temp_topics = _c_shadow[d][index-1];
        _z_shadow[d][index-1] = _z[d];
        _z[d] = temp;
        _c_shadow[d][index-1] = _c[d];
        _c[d] = temp_topics;
    } else {
        //VLOG(1) << "not swaping [" << _document_name[d] <<"]";
    }

    add_all_words_from(d, _c[d], _z[d]);
}

// Assume that all the words for document d have been assigned using the level
// assignment zd, now remove them all.
void GEMNCRPFixed::remove_all_words_from(unsigned d, vector<CRP*>& cd, WordToCountMap& zd) {
    // Remove this document's words from the relevant counts
    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];
        cd[zd[n]]->nw[w] -= 1;  // # of words in topic z equal to w
        cd[zd[n]]->nd[d] -= 1;  // # of words in doc d with topic z
        cd[zd[n]]->nwsum -= 1;  // # of words in topic z

        CHECK_LE(cd[zd[n]]->nw[w], _total_word_count);
        CHECK_LE(cd[zd[n]]->nwsum, _total_word_count);
    }
}

// Assume that all the words for document d have been removed, now add them
// back using the level assignment zd, now remove them all.
void GEMNCRPFixed::add_all_words_from(unsigned d, vector<CRP*>& cd, WordToCountMap& zd) {
    // Remove this document's words from the relevant counts
    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];
        cd[zd[n]]->nw[w] += 1;  // # of words in topic z equal to w
        cd[zd[n]]->nd[d] += 1;  // # of words in doc d with topic z
        cd[zd[n]]->nwsum += 1;  // # of words in topic z

        CHECK_LE(cd[zd[n]]->nw[w], _total_word_count);
        CHECK_LE(cd[zd[n]]->nwsum, _total_word_count);
    }
}

// Returns the (unnormalized) path probability for document d given the current
// set of _z assignments
double GEMNCRPFixed::compute_path_probability_for(unsigned d, vector<CRP*>& cd) {
    double lp_c_d = 0;

    for (unsigned l = 0; l < cd.size(); l++) {
        for (int k = 0; k < _D[d].size(); k++) {
            unsigned w = _D[d][k];
            lp_c_d += gammaln(cd[l]->nw[w] + _eta[w])
                - gammaln(cd[l]->nwsum + _eta_sum);
        }
    }

    return lp_c_d;
}



// If we're doing hyperparameter updates, then sample eta using Metropolis
// Hastings steps
void GEMNCRPFixed::resample_posterior_eta() {
    vector<double> new_eta;
    double new_eta_sum = 0;

    // Set up the new proposal and calculate B^{-K}(\beta) for both the new
    // distribution and the old one
    double lp_eta = 0;
    double lp_new_eta = 0;
    for (int v = 0; v < _eta.size(); v++) {
        //LOG(INFO) << _eta[v] << " + " << sample_gaussian();
        if (sample_uniform() < 0.005) {
            new_eta.push_back(max(0.0001, _eta[v] + sample_gaussian() / 10.0));
        } else {
            new_eta.push_back(_eta[v]);
        }
        new_eta_sum += new_eta[v];

        lp_eta -= gammaln(_eta[v]);
        lp_new_eta -= gammaln(new_eta[v]);
    }

    lp_eta += gammaln(_eta_sum);
    lp_new_eta += gammaln(new_eta_sum);

    double lp_eta2 = 0;
    double lp_new_eta2 = 0;

    // Multiply probabilities over all the topics
    deque<CRP*> node_queue;
    set<CRP*> visited;
    node_queue.push_back(_ncrp_root);
    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();

        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        for (WordToCountMap::iterator itr = current->nw.begin();
                itr != current->nw.end();
                itr++) {
            unsigned w = itr->first;  // the word
            unsigned count = itr->second;

            lp_eta2 += gammaln(_eta[w] + count);
            lp_new_eta2 += gammaln(new_eta[w] + count);
        }

        lp_eta2 -= gammaln(current->nwsum + _eta_sum);
        lp_new_eta2 -= gammaln(current->nwsum + new_eta_sum);
    }

    // With this, we have now computed up to B^{-|V|}(\beta)
    lp_eta *= visited.size();
    lp_new_eta *= visited.size();

    lp_eta += lp_eta2;
    lp_new_eta += lp_new_eta2;

    // Add in the prior (for now this is uniform)
    // XXX

    // Now check to see if we should MH step
    double k = log(sample_uniform());

    LOG(INFO) << "X " << lp_new_eta << " " << lp_eta << " = " << lp_new_eta - lp_eta << " " << new_eta_sum / (double)new_eta.size();
    if (k < lp_new_eta - lp_eta) {
        LOG(INFO) << "RESAMPLED";
        _eta = new_eta;
        _eta_sum = new_eta_sum;
    }
}


// Resamples the level allocation variables z_{d,n} given the path assignments
// c and the path assignments given the level allocations
void GEMNCRPFixed::resample_posterior() {
    CHECK_GT(_lV, 0);
    CHECK_GT(_lD, 0);
    CHECK_GT(_L, 0);

    if (FLAGS_learn_eta) {
        resample_posterior_eta();
    }

    // Interleaved version
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        VLOG(1) <<  "  resampling document " <<  d;
        if (!_c[d].empty()) {
            resample_posterior_z_for(d, true);
            if (FLAGS_sense_selection) {
                resample_posterior_c_for(d);
            }
        }
        // DCHECK(tree_is_consistent());
    }
}


double GEMNCRPFixed::compute_log_likelihood() {
    // Compute the log likelihood for the tree
    double log_lik = 0;

    // Compute the log likelihood of the level assignments (correctly?)
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        if (_c[d].empty()) {
            continue;
        }

        if (FLAGS_gem_sampler) {
            // TODO: inefficient?
            // ndsum_above[k] is #[z_{d,-n} >= k]
            vector<unsigned> ndsum_above;
            for (int l = 0; l < _c[d].size(); l++) {
                // TODO: optimize this
                ndsum_above.push_back(0);
                // count the total words attached for this document below here
                for (int ll = l; ll < _c[d].size(); ll++) {
                    ndsum_above[l] += _c[d][ll]->nd[d];
                }
                // LOG(INFO) << "ndsum_above[" << l << "] = " << ndsum_above[l];
            }

            for (int n = 0; n < _D[d].size(); n++) {
                // likelihood of drawing this word
                unsigned w = _D[d][n];
                log_lik += log(_c[d][_z[d][n]]->nw[w]+_eta[w]) -
                    log(_c[d][_z[d][n]]->nwsum+_eta_sum);

                // likelihood of the topic?
                // TODO: this is heinously inefficient
                double V_j_sum = 0;
                for (int l = 0; l < _z[d][n]; l++) {
                    if (l < _c[d].size()-1) {
                        V_j_sum += log(_gem_m*_pi + ndsum_above[l+1]) -
                            log(_pi + ndsum_above[l]);
                    }
                }

                log_lik += log((1-_gem_m)*_pi + _c[d][_z[d][n]]->nd[d]) -
                    log(_pi + ndsum_above[_z[d][n]]) + V_j_sum;
            }
        } else {  // multinomial sampler
            for (int n = 0; n < _D[d].size(); n++) {
                // likelihood of drawing this word
                unsigned w = _D[d][n];
                if (FLAGS_use_reject_option
                        && _c[d][_z[d][n]]->label.find("REJECT") == 0) {
                    // log_lik += -log(_D[d].size());  // uniform over document
                    log_lik += -log(_lV);  // uniform over vocab
                } else {
                    CHECK_LE(_c[d][_z[d][n]]->nw[w], _c[d][_z[d][n]]->nwsum);
                    log_lik += log(_c[d][_z[d][n]]->nw[w]+_eta[w]) -
                        log(_c[d][_z[d][n]]->nwsum+_eta_sum);
                    CHECK_LE(log_lik, 0) << "log likelihood went positive for [" << _document_name[d] << "]";
                }
                // likelihood of the topic?
                //  XXX: original wrong _L
                // log_lik += log(_c[d][_z[d][n]]->nd[d]+_alpha)
                //  - log(_nd[d]+_L*_alpha);
                log_lik += log(_c[d][_z[d][n]]->nd[d]+_alpha[_z[d][n]]/_c[d].size())
                    - log(_nd[d]+_alpha[_z[d][n]]);
                CHECK_LE(log_lik, 0);
            }
        }
    }
    return log_lik;
}
