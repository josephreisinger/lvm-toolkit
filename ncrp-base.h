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


#ifndef CRP_BASE_H_
#define CRP_BASE_H_

// TODO:: currently we have these depencies because I couldn't figure out
// how to (easily) write plain ascii files to the disk. One solution is to
// figure out how to do that, a better, more long-term friendly solution is to
// design a protocol buffer to store the output samples. This would be more
// space efficient, among other things.
#include <iostream>
#include <fstream>

#include <math.h>

#include <set>
//#include <hash_map>
#include <string>
#include <vector>
#include <string>
#include <deque>

#include "gibbs-base.h"

#include "strutil.h"

using namespace std;
using namespace google::protobuf;

// number of times to resample z (the level assignments) per iteration of c (the
// tree sampling)
DECLARE_int32(ncrp_z_per_iteration);

// this is the actual data file containing a list of attributes on each line,
// tab separated, with the first entry being the class label.
DECLARE_string(ncrp_datafile);

// Alpha controls the topic smoothing, with higher alpha causing more "uniform"
// distributions over topics. This is replaced by m and pi in the GEM sampler.
DECLARE_double(ncrp_alpha);


// Gamma controls the probability of creating new brances in both the
// Multinomial and GEM sampler; has no effect in the fixed-structure sampler.
DECLARE_double(ncrp_gamma);

// Setting this to true interleaves Metropolis-Hasting steps in between the
// Gibbs steps to update the hyperparameters. Currently it is only implemented
// in the basic version.
DECLARE_bool(ncrp_update_hyperparameters);

// Setting this to true causes the hyperparameter gamma to be scaled by m, the
// number of documents attached to the node. This makes branching into a
// constant proportion (roughly \gamma / (\gamma + 1)) indepedent of node size
// (slighly more intuitive behavior). If this isn't set, you're likely to get
// long chains instead of branches
DECLARE_bool(ncrp_m_dependent_gamma);

// This places an (artificial) cap on the number of branches possible from each
// node, reducing the width of the tree, but sacrificing the generative
// semantics of the model. -1 is the default for no capping.
DECLARE_int32(ncrp_max_branches);

// Parameter controlling the depth of the tree. Any interior node can have an
// arbitrary number of branches, but paths down to the leaves are constrained
// to be exactly this length.
DECLARE_int32(ncrp_depth);

// If set to true, don't assign any words to the root node; this still maintains
// the generative semantics of the model, but gives us a free implementation of
// the dirichlet process (L=2, skip root) and as well as mixture of ncrps.
DECLARE_bool(ncrp_skip_root);

// Setting this forces the topic topology to consist of a length L-1 chain
// followed by a set of leaves at the end. Basically the idea is to get a set of
// L-1 "noise" topics and a single "signal" topic; so this is really an
// implementation of prix-fixe with more than one noise.
DECLARE_bool(ncrp_prix_fixe);

// Eta depth scale: multiply eta by eta_depth_scale**depth for nodes at that
// depth; essentially eta_depth_scale=0.5 will lead to more mass at higher
// nodes, as opposed to leaves
DECLARE_double(ncrp_eta_depth_scale);

// The hLDA base class, contains code common to the Multinomial (fixed-depth)
// and GEM (infinite-depth) samplers
class NCRPBase : public GibbsSampler {
    public:
        // Initialization routines; must be called before run. Sets up the tree
        // and list views of the nCRP and does the initial level assignment. The
        // tree is grown incrementally from a single branch using the same
        // agglomerative method as resample_posterior_c. This procedure is
        // recommended by Blei.
        NCRPBase();
        virtual ~NCRPBase() { /* TODO: free memory! */ }

        // Allocate all the documents at once (called for non-streaming)
        void batch_allocation();

        // Allocate a single document; can be called during load for streaming
        void allocate_document(unsigned d);

        // Deallocate a random document from the model
        void deallocate_document();

        // Write out the learned tree in dot format
        virtual void write_data(string prefix);

        // return a string describing the current state of the hyperparameters
        // and log likelihood
        virtual string current_state() = 0;

    protected:
        virtual void resample_posterior_z_for(unsigned d, bool remove) = 0;
        void resample_posterior_c_for(unsigned d);

        void calculate_path_probabilities_for_subtree(CRP* root,
                unsigned d,
                unsigned max_depth,
                LevelWordToCountMap& nw_removed,
                LevelToCountMap& nwsum_removed,
                vector<double>* lp_c_d,
                vector<CRP*>* c_d);

        virtual double compute_log_likelihood() = 0;

        bool tree_is_consistent();  // check the consistency of the tree

        // Returns a list of nodes in this path. If node is internal, then it grows
        // down to depth L. If node is a leaf, then it just returns the path to the
        // root
        void graft_path_at(CRP* node, vector<CRP*>* chain, unsigned depth);

        void print_summary();

    protected:

        // Parameters
        unsigned _L;

        // eta = "scale" of the topic model, higher eta = more general / fewer topics.
        vector<double> _alpha;
        double _gamma;  // hyperparams

        double _alpha_sum;  // normalization constants

        DocWordToCount  _z;  // level assignments per document, word
        DocToTopicChain _c;  // CRP nodes for a document m

        CRP* _ncrp_root;  // tree representation of the nCRP.
        CRP* _reject_node;  // special node containing rejected attributes.

        unsigned _unique_nodes;  // number of leaves in the tree

        string _filename;  // output file name

        unsigned _total_words;  // total number of words added
};

#endif  // CRP_BASE_H_
