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
// This file contains all of the common code shared between the Mutinomial hLDA
// sampler (fixed-depth tree) and the GEM hLDA sampler (infinite-depth tree), as
// well as the fixed topic structure sampler used to learn WN senses.
//
// CRPNode is the basic data structure for a single node in the topic tree (dag)

#include <string>
#include <fstream>

#include "dSFMT-src-2.0/dSFMT.h"

#include "gibbs-base.h"
#include "ncrp-base.h"

using namespace std;


// number of times to resample z (the level assignments) per iteration of c (the
// tree sampling)
DEFINE_int32(ncrp_z_per_iteration,
             1,
             "Number of times to update the word-topic assignments per path allocation.");

// this is the actual data file containing a list of attributes on each line,
// tab separated, with the first entry being the class label.
DEFINE_string(ncrp_datafile,
              "",
              "the input data set, words arranged in documents");

// Alpha controls the topic smoothing, with higher alpha causing more "uniform"
// distributions over topics. This is replaced by m and pi in the GEM sampler.
DEFINE_double(ncrp_alpha,
              1.0,
              "hyperparameter alpha, controls the topic smoothing");


// Gamma controls the probability of creating new brances in both the
// Multinomial and GEM sampler; has no effect in the fixed-structure sampler.
DEFINE_double(ncrp_gamma,
              0.1,
              "hyperparameter gamma, controls the tree smoothing");

// Setting this to true interleaves Metropolis-Hasting steps in between the
// Gibbs steps to update the hyperparameters. Currently it is only implemented
// in the basic version.
DEFINE_bool(ncrp_update_hyperparameters,
            false,
            "should the hyperparameters be learned as well?");

// Setting this to true causes the hyperparameter gamma to be scaled by m, the
// number of documents attached to the node. This makes branching into a
// constant proportion (roughly \gamma / (\gamma + 1)) indepedent of node size
// (slighly more intuitive behavior). If this isn't set, you're likely to get
// long chains instead of branches
DEFINE_bool(ncrp_m_dependent_gamma,
            false,
            "should gamma depend on m?");

// This places an (artificial) cap on the number of branches possible from each
// node, reducing the width of the tree, but sacrificing the generative
// semantics of the model. -1 is the default for no capping.
DEFINE_int32(ncrp_max_branches,
             -1,
              "maximum number of branches from any node");

// Parameter controlling the depth of the tree. Any interior node can have an
// arbitrary number of branches, but paths down to the leaves are constrained
// to be exactly this length.
DEFINE_int32(ncrp_depth, 5, "tree depth (L)");

// If set to true, don't assign any words to the root node; this still maintains
// the generative semantics of the model, but gives us a free implementation of
// the dirichlet process (L=2, skip root) and as well as mixture of ncrps.
DEFINE_bool(ncrp_skip_root,
            false,
            "Don't assign anything to the root node; ignore it as a topic.");

// Setting this forces the topic topology to consist of a length L-1 chain
// followed by a set of leaves at the end. Basically the idea is to get a set of
// L-1 "noise" topics and a single "signal" topic; so this is really an
// implementation of prix-fixe with more than one noise.
DEFINE_bool(ncrp_prix_fixe,
            false,
            "implements prix-fixe clustering with L-1 noise topics");


// Initialize the NCRPBase tree by adding each document (set of attributes)
// incrementaly, resampling level (tree) assignments after each document is
// added
NCRPBase::NCRPBase() 
    : _L(FLAGS_ncrp_depth), _gamma(FLAGS_ncrp_gamma), _reject_node(NULL) {
    LOG(INFO) << "initialize ncrp_base";

    // Initialize the per-topic dirichlet parameters
    // NOTE: in reality this would actually have to be /per topic/ as in one
    // parameter per node in the hierarchy. But since its not used for now its
    // ok.
    for (int l = 0; l < _L; l++) {
        _alpha.push_back(FLAGS_ncrp_alpha);
    }
    _alpha_sum = _L*FLAGS_ncrp_alpha;

    _ncrp_root = new CRP(0, 0);

    // Allocate the first chain of L guys
    CRP* current = _ncrp_root;
    for (unsigned l = 1; l < _L; l++) {
        current->tables.push_back(new CRP(l, 0, current));
        current = current->tables[0];
    }

    // For each document, allocate a topic path for it there are several ways to
    // do this, e.g. a single linear chain, incremental conditional sampling and
    // random tree
    _c.set_empty_key(kEmptyUnsignedKey); 
    _z.set_empty_key(kEmptyUnsignedKey); 
    _c.set_deleted_key(kDeletedUnsignedKey); 
    _z.set_deleted_key(kDeletedUnsignedKey); 
}

void NCRPBase::batch_allocation() {
    LOG(INFO) << "Doing batch allocation...";


    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;
        allocate_document(d);
    }
}


void NCRPBase::allocate_document(unsigned d) {
    // Initially assign the document to a random branch (these counts will
    // be removed immediately during the resample step)
    for (int l = 0; l < _L; l++) {
        if (l == 0) {
            _c[d].push_back(_ncrp_root);
        } else {
            _c[d].push_back(_c[d][l-1]->tables[0]);
        }
        CHECK(_c[d][l]);
        _c[d][l]->ndsum += 1;  // number of docuemnts in this CRP
    }

    // Incrementally reconstruct the tree (each time we add a document,
    // update its tree assignment based only on the previously added
    // documents; this results in a "fuller" initial tree, instead of one
    // fat trunk (fat trunks cause problems for mixing)
    if (d == 0) {
        // Initial uniform random level assignments
        for (int n = 0; n < _D[d].size(); n++) {
            unsigned w = _D[d][n];

            // set a random topic assignment for this guy
            _z[d][n] = FLAGS_ncrp_skip_root ? sample_integer(_L-1)+1 : sample_integer(_L);

            // test the initialization of maps
            CHECK(_c[d][_z[d][n]]->nw.find(w) != _c[d][_z[d][n]]->nw.end()
                    || _c[d][_z[d][n]]->nw[w] == 0);
            CHECK(_c[d][_z[d][n]]->nd.find(d) != _c[d][_z[d][n]]->nd.end()
                    || _c[d][_z[d][n]]->nd[d] == 0);

            // Can't use add b/c it doesn't respect the fact that we can have
            // interior nodes with no words assigned
            // _c[d][_z[d][n]]->add(w,d);
            _c[d][_z[d][n]]->add_no_ndsum(w,d);
        }
    } else {
        resample_posterior_z_for(d, false);  // false means we don't remove the document first, adding it

        if (FLAGS_ncrp_depth > 1 && FLAGS_ncrp_max_branches != 1) {
            resample_posterior_c_for(d);
        }
    }

    if (d % 1000 == 0 && d > 0) {
      LOG(INFO) << "Sorted " << d << " documents into " << _unique_nodes << " clusters.";
    }
}

void NCRPBase::deallocate_document() {
    // This block chooses a random guy from the map
    unsigned cut = sample_integer(_lD);
    unsigned i = 0;
    unsigned d;
    for (DocumentMap::const_iterator d_itr = _D.begin();
            d_itr != _D.end(); d_itr++, i++) {
        if (i == cut) {
            d = d_itr->first;
            break;
        }
    }

    VLOG(1) << "deallocating " << d;

    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];
        _c[d][_z[d][n]]->remove(w,d);
    }
    _z.erase(d);
    _c.erase(d);
    _D.erase(d);
    _lD = _D.size();
}


// Resamples the tree given the level allocation variables for each document
// conditional on z
void NCRPBase::resample_posterior_c_for(unsigned d) {
    CHECK_EQ(FLAGS_streaming, 0) << "this hasn't been prepped for deallocation";
    VLOG(1) << "resample posterior c for " << d;
    LevelWordToCountMap nw_removed;
    LevelToCountMap     nwsum_removed;
    nw_removed.set_empty_key(kEmptyUnsignedKey); 
    nwsum_removed.set_empty_key(kEmptyUnsignedKey); 

    // Remove this document's words from the relevant counts
    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];

        _c[d][_z[d][n]]->remove_no_ndsum(w,d);
        // Keep track of the removed counts for computing the likelihood of
        // the data
        nw_removed[_z[d][n]][w] += 1;
        nwsum_removed[_z[d][n]] += 1;
    }

    // Remove this document from the tree
    for (int l = 0; l < _c[d].size(); l++) {
        CHECK_EQ(_c[d][l]->nd[d], 0);

        _c[d][l]->ndsum -= 1;
        CHECK_GE(_c[d][l]->ndsum, 0);
    }
    // This next bit assumes _c[d] is ordered by level?
    /*for (int l = 0; l < _c[d].size(); l++) {
        if (_c[d][l]->ndsum == 0) {  // kill this entire branch
          LOG(INFO) << "XXX " << d << " l=" << l;
            LOG(INFO) << "about to delete " << d << " " << _c[d][l];
            // remove ourselves from the previous's next pointers
            delete _c[d][l];  // this will recurse through the children
            break;
        }
    }*/


    // Run over the entire tree level by level to compute the probability
    // for each new assignment
    vector<double> lp_c_d;  // log-probability of this branch c_d
    vector<CRP*> c_d;  // the actual branch c_d

    calculate_path_probabilities_for_subtree(_ncrp_root, d, _c[d].size(), nw_removed, nwsum_removed, &lp_c_d, &c_d);

    // Actually do the sampling (select a new path)
    // int index = SAFE_sample_unnormalized_log_multinomial(&lp_c_d);
    int index = sample_unnormalized_log_multinomial(&lp_c_d);

    // Update d's path
    graft_path_at(c_d[index], &_c[d], _c[d].size());

    // Restore the document counts too
    for (int l = 0; l < _c[d].size(); l++) {
        _c[d][l]->ndsum += 1;
        CHECK_GT(_c[d][l]->ndsum, 0);
    }

    // Add back in document D_d
    for (int n = 0; n < _D[d].size(); n++) {
        unsigned w = _D[d][n];
        _c[d][_z[d][n]]->add_no_ndsum(w,d);
    }
    VLOG(1) << "done";
}

// Starting from node root, calculate the probability of attaching document d
// down any possible subtree (including new ones that might be added) to a
// maximum depth of _c[d]'s required depth
void NCRPBase::calculate_path_probabilities_for_subtree(
        CRP* root,
        unsigned d,
        unsigned max_depth,
        LevelWordToCountMap& nw_removed,
        LevelToCountMap&     nwsum_removed,
        vector<double>* lp_c_d,
        vector<CRP*>* c_d) {
    // Loop over every node in the tree using a level-by-level traversal,
    // ensuring that each time we visit a node, we have already calculated the
    // path probability up to its parent
    VLOG(2) << "Calculate path probabilities";
    _unique_nodes = 0;  // recalculate the tree size
    deque<CRP*> node_queue;
    node_queue.push_back(root);
    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();

        CHECK(current);
        _unique_nodes += 1;

        // TODO:: this is only good for the infinite depth version, the
        // fixed depth version needs the check above
        // CHECK_GT(current->ndsum, 0) << "should not be summing over empty trees";
        if (current->ndsum == 0) {
            CHECK(current != root) << "tried to delete the root!";
            // LOG(INFO) << "about to delete " << d << " " << current;
            delete current;  // this will recurse through the children
            continue;
        }

        // Take care of the current level probabilities
        if (current->prev.size() > 0) {  // not the root
            // the standard nCRP doesn't work on DAGs, only trees
            CHECK_EQ(current->prev.size(), 1);

            // compute the probability of getting to this node
            if (FLAGS_ncrp_m_dependent_gamma) {
                current->lp = log(current->ndsum) - log((_gamma + 1) * current->prev[0]->ndsum - 1) + current->prev[0]->lp;
            } else {
                current->lp = log(current->ndsum) - log(_gamma + current->prev[0]->ndsum - 1) + current->prev[0]->lp;
            }
        } else {
            current->lp = 0;
        }

        // multiply in the data likelihood for this level
        // Compute a single level's contribution to the log data likelihood
        current->lp += gammaln(current->nwsum + _eta_sum)
            - gammaln(current->nwsum + nwsum_removed[current->level]
                    + _eta_sum);

        // We don't care about the terms here where nw_removed is zero, since
        // they cancel out.
        for (WordToCountMap::iterator itr = nw_removed[current->level].begin(); itr != nw_removed[current->level].end(); itr++) {
            unsigned w = itr->first;  // the word
            unsigned count = itr->second;
            current->lp += gammaln(current->nw[w] + count + _eta[w]) - gammaln(current->nw[w] + _eta[w]);
        }

        // Now the rest of the vocabulary is accounted for, since
        // gammaln(0+0+eta) - gammaln(0+eta) = 0

        // Now start adding in the next level stuff
        // If prix-fixe is turned on, then we should only branch if we're at the
        // second-to-last level (max_depth-2) otherwise if prix-fixe is turned
        // off we can just brach as lnog as we're not the max level
        if (current->level < max_depth-1) {
            // i.e. internal to this topic chain (_c[d]), so might make a new
            // branch
            if ((!FLAGS_ncrp_prix_fixe || current->level == max_depth-2) 
                    && (FLAGS_ncrp_max_branches == -1 || current->tables.size() < FLAGS_ncrp_max_branches)) {
                // Add the probability of escaping from this node (new branch)

                // Base log-probability of getting here plus taking the new table
                double prob = 0;
                if (FLAGS_ncrp_m_dependent_gamma) {
                    // NOTE before this was:
                    // prob = current->lp + log(_gamma * current->ndsum) - log(_gamma + current->ndsum - 1);
                    prob = current->lp + log(_gamma * current->ndsum) - log((_gamma+1) * current->ndsum - 1);
                } else {
                    prob = current->lp + log(_gamma) - log(_gamma + current->ndsum - 1);
                }


                // Add in the log-data-likelihood all the way down the new chain
                // taking into account this document's current chain length
                for (int l = current->level+1; l < _c[d].size(); l++) {
                    prob += gammaln(_eta_sum) - gammaln(nwsum_removed[l] + _eta_sum);

                    int total_removed = 0;

                    // This is actually computing over w \in V but when count=0 the etas
                    // cancel
                    for (WordToCountMap::iterator itr = nw_removed[l].begin(); itr != nw_removed[l].end(); itr++) {
                        // itr->first = word
                        // itr->second = count
                        prob += gammaln(itr->second + _eta[itr->first]) - gammaln(_eta[itr->first]);
                        total_removed += itr->second;
                    }
                    CHECK_EQ(total_removed, nwsum_removed[l]);
                }

                lp_c_d->push_back(prob);
                c_d->push_back(current);
            }

            node_queue.insert(node_queue.end(), current->tables.begin(),
                    current->tables.end());
        } else {
            // Add the probability of reaching this node (old branch)
            lp_c_d->push_back(current->lp);
            c_d->push_back(current);
        }
    }
    VLOG(2) << "done";
}

// Returns the list of nodes in the path containing node extending to depth
// depth.. If node is internal, then it grows down upto depth depth. If node is
// a leaf, then it just returns the path to the root
void NCRPBase::graft_path_at(CRP* node, vector<CRP*>* chain, unsigned depth) {
    // Add back in the new path ; rebuild c First add in the head (everything up
    // to here)

    CHECK(node);
    // CHECK_EQ(chain->size(), 0);
    chain->clear();

    CRP* current = node;
    while (true) {
        CHECK_LE(current->prev.size(), 1);  // currently this only works for trees
        chain->insert(chain->begin(), current);
        if (current->prev.empty()) {
            break;
        }
        current = current->prev.at(0);
    }

    // CHECK(!node->tables.empty() || chain->size() == depth); // only valid for
    // the fixed depth version

    // Add new nodes, fleshing the chain out to depth depth
    if (node->level < depth) {
        current = node;
        for (int l = node->level+1; l < depth; l++) {
            // create a new chain of restaurants
            CRP* new_crp = new CRP(l, 0, current);  // add back pointer
            current->tables.push_back(new_crp);  // add forward pointer
            // LOG(INFO) << "r " << current->tables.size();
            CHECK(FLAGS_ncrp_max_branches == -1 || current->tables.size() <= FLAGS_ncrp_max_branches);
            chain->push_back(new_crp);

            current = new_crp;  // iterate
        }
        CHECK_EQ(chain->size(), depth);
    }
}



// Write out all the data in an intermediate format
void NCRPBase::write_data(string prefix) {
    // File* f = File::OpenOrDie(filename, "w");
    // RecordWriter f (file);   // Create RecordWriter on the file.

    // f->Write(string("digraph G {\n"));

    // Close the file, checking error code.
    // CHECK(f->Close());
    string filename = StringPrintf("%s-%d-%s.hlda", get_base_name(_filename).c_str(), FLAGS_random_seed,
            prefix.c_str());

    ofstream f(filename.c_str(), ios_base::out | ios_base::binary);

    f << current_state() << endl;

    set<CRP*> visited_nodes;  // keep track of visited things for the DAG case
    deque<CRP*> node_queue;
    node_queue.push_back(_ncrp_root);
    if (_reject_node) {
        node_queue.push_back(_reject_node);
    }
    // check for empty nodes
    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();

        if (visited_nodes.find(current) == visited_nodes.end()) {
            f << current << "\t||\t" << current->label << "\t||\t" << current->ndsum
                << "\t||";

            for (WordToCountMap::iterator nw_itr = current->nw.begin();
                    nw_itr != current->nw.end(); nw_itr++) {
                if (nw_itr->second > 0) {  // sparsify
                    f << "\t" << _word_id_to_name[nw_itr->first] << "@@@" << nw_itr->second;
                }
            }
            f << "\t||";
            for (DocToWordCountMap::iterator nd_itr = current->nd.begin();
                    nd_itr != current->nd.end(); nd_itr++) {
                if (nd_itr->second > 0) {  // sparsify
                    f << "\t" << nd_itr->first << "@@@" << _document_name[nd_itr->first]
                        << "@@@" << nd_itr->second;
                }
            }
            f << "\t||";
            for (int i = 0; i < current->tables.size(); i++) {
                f << "\t" << current->tables[i];
            }
            f << endl;

            node_queue.insert(node_queue.end(), current->tables.begin(),
                    current->tables.end());
            visited_nodes.insert(current);
        }
    }
}


// Check to see if the tree view starting at ncrp_root is consistent with the
// list view starting at each of the _c values.
bool NCRPBase::tree_is_consistent() {
    LOG(INFO) << "bad bad";
    vector<CRP*> visited;
    deque<CRP*> node_queue;
    node_queue.push_back(_ncrp_root);

    unsigned total_words = 0;

    // check for empty nodes
    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();

        CHECK(current);  // node exists
        // node has documents
        CHECK_GT(current->ndsum, 0) << "node [" << current->label << "] has no docs.";

        for (WordToCountMap::iterator itr = current->nw.begin();
                itr != current->nw.end(); itr++) {
            total_words += itr->second;
        }

        // Check that the number of documents attached to the children of this
        // node is equal to the number of documents attached to this node.
        // TODO:: this is only for the fixed depth version, figure out a way to
        // include it
        // if (!current->tables.empty()) {
        //    int sum = 0;
        //    for (int i = 0; i < current->tables.size(); i++) {
        //        sum += current->tables[i]->ndsum;
        //    }
        //    CHECK_EQ(sum, current->ndsum);
        //
        //    // Check that the root sees the correct number of documents.
        //    // Correcting for whether we have removed one or not.
        //    if (current == _ncrp_root) {
        //        CHECK_EQ(sum,_lD);
        //    }
        // }



        // check that at least m documents in _c point here
        int count = 0;
        for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
            unsigned d = d_itr->first;

            // CHECK( _L == 0 || _L == _c[d].size() );
            for (int l = 0; l < _c[d].size(); l++) {
                if (_c[d][l] == current) {
                    count += 1;
                }
            }
        }

        CHECK_EQ(count, current->ndsum);

        visited.push_back(current);

        node_queue.insert(node_queue.end(), current->tables.begin(),
                current->tables.end());
    }

    CHECK_EQ(total_words, _total_word_count);

    // Check we visited everything
    for (DocumentMap::const_iterator d_itr = _D.begin(); d_itr != _D.end(); d_itr++) {
        unsigned d = d_itr->first;

        for (int l = 0; l < _c[d].size(); l++) {
            // LOG(INFO) << "_c[" << d << "][" << l << "] = " << _c[d][l];
            CHECK(_c[d][l]);
            CHECK(find(visited.begin(), visited.end(), _c[d][l]) != visited.end());

            // make sure the next guy in the list is actually a child in the
            // tree
            if (l < _c[d].size()-1) {
                CHECK(find(_c[d][l]->tables.begin(), _c[d][l]->tables.end(), _c[d][l+1])
                        != _c[d][l]->tables.end());
            }
            // likewise make sure the previous
            if (l > 0) {
                CHECK(find(_c[d][l-1]->tables.begin(), _c[d][l-1]->tables.end(),
                            _c[d][l]) != _c[d][l-1]->tables.end());
            }
        }
    }
    return true;
}

// Prints out the top few features from each cluster
void  NCRPBase::print_summary() {
    // Loop over every node in the tree using a level-by-level traversal,
    // ensuring that each time we visit a node, we have already calculated the
    // path probability up to its parent
    deque<CRP*> node_queue;
    node_queue.push_back(_ncrp_root);
    unsigned l = 0;
    while (!node_queue.empty()) {
        CRP* current = node_queue.front();
        node_queue.pop_front();

        CHECK(current);

        string buffer = show_chopped_sorted_nw(current->nw);
        if (current->level != _L-1) {
            // Convert the ublas vector into a vector of pairs for sorting
            LOG(INFO) << "N[" << current->level << "] (" << StringPrintf("%.3f\%", current->nwsum / (double)_total_word_count)  
                << ") " << " " << buffer;
        } else {
            LOG(INFO) << "C[" << l << "] (" << current->ndsum << ") " << " " << buffer;
            l += 1;
        }

        node_queue.insert(node_queue.end(), current->tables.begin(),
                                    current->tables.end());
    }

}
