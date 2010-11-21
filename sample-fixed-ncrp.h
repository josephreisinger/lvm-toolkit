// Samples from the Nested Chinese Restaurant Process using a fixed topic
// structure. This model is more expressive in that the topic structure is not
// constrained to be a tree, but rather a digraph.

#ifndef SAMPLE_GEM_FIXED_NCRP_H_
#define SAMPLE_GEM_FIXED_NCRP_H_

#include <string>
#include <vector>

#include "ncrp-base.h"

// The GEM(m, \pi) distribution hyperparameter m, controls the "proportion of
// general words relative to specific words"
DECLARE_double(gem_m);

// The GEM(m, \pi) hyperparameter \pi: reflects how strictly we expect the
// documents to adhere to the m proportions.
DECLARE_double(gem_pi);

// The file path from which to load the topic structure. The file must be
// encoded as one connection per line, child <tab> parent.
DECLARE_string(tree_structure_file);

// Whether or not to use the GEM sampler. The Multinomial sampler currently is
// more flexible as it allows the tree structure to be a DAG; the GEM sampler
// might not work yet with DAGs.
DECLARE_bool(gem_sampler);

// If unset, then just throw away extra edges that cause nodes to have multiple
// parents. Enforcing a tree topology.
DECLARE_bool(use_dag);

// Should non-WN class nodes have words assigned to them? If not, then all
// topics will start with wn_
DECLARE_bool(fold_non_wn);

// Should we perform variable selection (i.e. attribute rejection) based on
// adding a single "REJECT" node with a uniform distribution over the
// vocabulary to each topic list?
DECLARE_bool(use_reject_option);

// Should the hyperparameters on the vocabulary Dirichlet (eta) be learned. For
// now this uses moment matchin to perform the updates.
DECLARE_bool(learn_eta);

// Should all the path combinations to the root be separated out into different
// documents? DAG only.
DECLARE_bool(separate_path_assignments);

// Should we try to learn a single best sense from a list of senses?
DECLARE_bool(sense_selection);

typedef google::dense_hash_map<unsigned, DocToTopicChain> DocSenseToTopicChain;
typedef google::dense_hash_map<unsigned, google::dense_hash_map<unsigned, DocToWordCountMap> > DocSenseWordToCount;

typedef google::dense_hash_map<unsigned, google::dense_hash_map<CRP*,double> > NodeLogFrequencyMap;

// This version differs from the normal GEM sampler in that the tree structure
// is fixed a priori. Hence there is no resampling of c, the path allocations.
class GEMNCRPFixed : public NCRPBase {
    public:
        GEMNCRPFixed(double m, double pi);
        ~GEMNCRPFixed() { /* TODO: free memory! */ }

        string current_state();

        void load_tree_structure(const string& filename);
        void load_precomputed_tree_structure(const string& filename);

    protected:
        void resample_posterior();
        void resample_posterior_z_for(unsigned d, bool remove) { resample_posterior_z_for(d, _c[d], _z[d]); }
        void resample_posterior_z_for(unsigned d, vector<CRP*>& cd, WordToCountMap& zd);
        void resample_posterior_c_for(unsigned d); // used in sense selection
        void resample_posterior_eta();

        double compute_log_likelihood();

        void contract_tree();

        void build_path_assignments(CRP* node, vector<CRP*>* c, int sense_index);
        void build_separate_path_assignments(CRP* node, vector< vector<CRP*> >* paths);

        // Assume that all the words for document d have been assigned using the level
        // assignment zd, now remove them all.
        void remove_all_words_from(unsigned d, vector<CRP*>& cd, WordToCountMap& zd);

        // Assume that all the words for document d have been removed, now add them
        // back using the level assignment zd, now remove them all.
        void add_all_words_from(unsigned d, vector<CRP*>& cd, WordToCountMap& zd);

        // Returns the (unnormalized) path probability for document d given the current
        // set of _z assignments
        double compute_path_probability_for(unsigned d, vector<CRP*>& cd);


    protected:
        double _gem_m;
        double _pi;

        // These hold other possible sense attachments that are not currently
        // in use.
        DocSenseWordToCount  _z_shadow;  // level assignments per document, word
        DocSenseToTopicChain _c_shadow;  // CRP nodes for a document m

        NodeLogFrequencyMap _log_node_freq;  // Gives the frequency of a sense attachment

        unsigned _maxL;
};

#endif  // SAMPLE_GEM_FIXED_NCRP_H_
