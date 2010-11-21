// Contains basic statistics routines and data structures for a discrete Gibbs
// sampler over some kind of document data.

#ifndef GIBBS_BASE_H_
#define GIBBS_BASE_H_

#include <iostream>
#include <fstream>

#include <math.h>

#include <limits.h>

#include <set>
#include <map>
#include <string>
#include <vector>
#include <string>
#include <deque>

#include <map>

#include <google/sparse_hash_map>
#include <google/dense_hash_map>

#include <glog/logging.h>
#include <gflags/gflags.h>


#include "strutil.h"

using namespace std;
using namespace google::protobuf;

// Streaming makes some major structural changes to the code, moving the main loop into load_data;
// This variable sets how many documents should be kept in memory at any one time
DECLARE_int32(streaming);

// Eta controls the amount of smoothing within the per-topic word distributions.
// Higher eta = more smoothing. Also used in the GEM sampler.
DECLARE_double(eta);

// We can incorporate prior ranking information via eta. One way to do this is
// to assume that eta is proportional to some exponential of the average word
// rank, thus more highly ranked words have higher priors. eta_prior_scale
// represents the degree of confidence (how fast to decay) in each rank, higher
// meaning less informative prior.
DECLARE_double(eta_prior_exponential);

// the number of gibbs iterations to run for.
DECLARE_int32(max_gibbs_iterations);

// One problem with Gibbs sampling is that nearby samples are highly
// correlated, throwing off the empirical distribution. In practice you need to
// wait some amount of time before reading each (independent) sample.
DECLARE_int32(sample_lag);

// The random seed
DECLARE_int32(random_seed);

// Our tolerance for numerical (under)overflow.
DECLARE_double(epsilon_value);

// How many steps of not writing a best do we need before we declare
// convergence?
DECLARE_int32(convergence_interval);

class CRP;

typedef google::sparse_hash_map<unsigned, unsigned> WordToCountMap;
typedef google::sparse_hash_map<unsigned, unsigned> DocToWordCountMap;
typedef google::dense_hash_map<unsigned, vector<CRP*> > DocToTopicChain;
typedef google::dense_hash_map<unsigned, unsigned> Docsize;
typedef google::dense_hash_map<unsigned, WordToCountMap> DocWordToCount;

typedef vector<unsigned> Document;

typedef google::dense_hash_map<unsigned, Document> DocumentMap;
typedef google::dense_hash_map<unsigned, string> DocIDToTitle;
typedef google::dense_hash_map<string, unsigned> TitleToDocID;

typedef google::dense_hash_map<unsigned, string> WordCode;


typedef google::dense_hash_map<unsigned, WordToCountMap > LevelWordToCountMap;
typedef google::dense_hash_map<unsigned, unsigned>        LevelToCountMap;

typedef google::dense_hash_map<unsigned,unsigned> cluster_map;
typedef map<unsigned,cluster_map> multiple_cluster_map;
typedef google::dense_hash_map<unsigned,CRP> clustering;
typedef map<unsigned,clustering> multiple_clustering;

typedef google::dense_hash_map<unsigned, WordToCountMap> topic_map;

const string kEmptyStringKey = "$$$EMPTY$$$";
const unsigned kEmptyUnsignedKey = UINT_MAX;
const unsigned kDeletedUnsignedKey = UINT_MAX-1;
const string kDeletedStringKey = "$$$DELETED$$$";

// A single node in the nCRP, corresponds to a table and also contains a list of
// children, e.g. the tables in the restaurant that it points to.
class CRP {
    public:
        CRP() : nwsum(0), label(""), ndsum(0) { 
            nw.set_deleted_key(kDeletedUnsignedKey); 
            nd.set_deleted_key(kDeletedUnsignedKey); 
        }
        CRP(unsigned l, unsigned customers)
            : level(l), nwsum(0), lp(0), label(""), ndsum(customers) { 
                nw.set_deleted_key(kDeletedUnsignedKey); 
                nd.set_deleted_key(kDeletedUnsignedKey); 
                // nw.set_empty_key(kEmptyUnsignedKey); 
                // nd.set_empty_key(kEmptyUnsignedKey); 
            }
        CRP(unsigned l, unsigned customers, CRP* p)
            : level(l), nwsum(0), lp(0), label(""), ndsum(customers) {
                prev.push_back(p); 
                nw.set_deleted_key(kDeletedUnsignedKey); 
                nd.set_deleted_key(kDeletedUnsignedKey); 
                // nw.set_empty_key(kEmptyUnsignedKey); 
                // nd.set_empty_key(kEmptyUnsignedKey); 
            }
        ~CRP();

        // Update ndsum to reflect the actual document assignments
        void add(unsigned w, unsigned d) {
            add_no_ndsum(w,d);

            if (nd[d] == 1) { // Added from a new doc
                ndsum += 1;
            }
        }
        void add_no_ndsum(unsigned w, unsigned d) {
            nw[w] += 1;
            nwsum += 1;
            nd[d] += 1;
        }

        void remove(unsigned w, unsigned d) {
            remove_no_ndsum(w,d);

            if (nd[d] == 0) { // Added from a new doc
                nd.erase(d);
                ndsum -= 1;
            }

            CHECK_GE(ndsum, 0);
        }

        void remove_no_ndsum(unsigned w, unsigned d) {
            nw[w] -= 1;
            nwsum -= 1;
            nd[d] -= 1;
            CHECK_GE(nw[w], 0);
            CHECK_GE(nwsum, 0);
            CHECK_GE(nd[d], 0);
            if (nw[w] == 0) {
                nw.erase(w);
            }
        }

        void remove_doc(const Document& D, unsigned d) {
            for (int n = 0; n < D.size(); n++) {
                unsigned w = D.at(n);
                // Remove this document and word from the counts
                remove(w,d);
            }
        }

        void add_doc(const Document& D, unsigned d) {
            for (int n = 0; n < D.size(); n++) {
                unsigned w = D.at(n);
                // Remove this document and word from the counts
                add(w,d);
            }
        }


        void remove_from_parents();

    public:
        WordToCountMap    nw;  // number of words equal to w in this node
        DocToWordCountMap nd;  // number of words from doc d in this node

        unsigned level;

        unsigned nwsum;  // number of words in this node
        unsigned ndsum;  // number of docs in this node (same as m)

        vector<CRP*> prev;  // the parents of this node in the DAG

        vector<CRP*> tables;  // the tables in the next restaurant

        double lp;  // probability of reaching this node, used in posterior_c

        // the node label from WN or whatever hierarchy (defaults to none)
        string label;
};

// The hLDA base class, contains code common to the Multinomial (fixed-depth)
// and GEM (infinite-depth) samplers
class GibbsSampler {
    public:
        // Initialization routines; must be called before run. Sets up the tree
        // and list views of the nCRP and does the initial level assignment. The
        // tree is grown incrementally from a single branch using the same
        // agglomerative method as resample_posterior_c. This procedure is
        // recommended by Blei.
        GibbsSampler() { 
            _unique_word_count = 0;
            _total_word_count = 0;
            _lD = 0;
            _lV = 0;

            _iter = 0;
            _best_ll = 0;
            _best_iter = 0;

            _converged_iterations = 0;

            _D.set_empty_key(kEmptyUnsignedKey);
            _D.set_deleted_key(kDeletedUnsignedKey);
            _word_name_to_id.set_empty_key(kEmptyStringKey);
            _word_id_to_name.set_empty_key(kEmptyUnsignedKey);
            _document_name.set_empty_key(kEmptyUnsignedKey); 
            _document_id.set_empty_key(kEmptyStringKey);
            _nd.set_empty_key(kEmptyUnsignedKey);
        }
        virtual ~GibbsSampler() { /* TODO: free memory! */ }

        // Allocate all documents at once
        virtual void batch_allocation() {
            LOG(FATAL) << "NYI batch_allocation";
        }

        // Allocate a document (called right after the document is read from the file)
        virtual void allocate_document(unsigned d) {
            LOG(FATAL) << "NYI allocate_document";
        }
        // Deallocate a document to conserve memory
        virtual void deallocate_document() {
            LOG(FATAL) << "NYI deallocate_document";
        }

        // Allocate new_d into the model and (maybe) kick out an old document
        void streaming_step(unsigned new_d);

        // Check log-likelihood and write out some samples if necessary
        bool sample_and_check_for_convergence();

        // Run the Gibbs sampler for max_gibbs_iterations iterations. This is the main
        // workhorse loop.
        void run();

        // Load a data file
        virtual void load_data(const string& filename);
        
        // Process a single document line from the file
        void process_document_line(const string& curr_line, unsigned line_no);

        // Write some summary of the output
        virtual void write_data(string prefix) = 0;

        // return a string describing the current state of the hyperparameters
        // and log likelihood
        virtual string current_state() = 0;

        string show_chopped_sorted_nw(const WordToCountMap& nw);

    protected:
        virtual void resample_posterior() = 0;

        virtual double compute_log_likelihood() = 0;

    protected:
        WordToCountMap    _V;  // vocabulary keys mapping to corpus counts
        unsigned _lV;  // size of vocab
        unsigned _lD;  // number of documents

        unsigned _unique_word_count;
        unsigned _total_word_count;

        WordCode _word_id_to_name;  // uniqe_id to string
        google::dense_hash_map<string, unsigned> _word_name_to_id;

        DocumentMap _D;  // documents indexed by unique #

        DocIDToTitle _document_name;  // doc_number to title
        TitleToDocID _document_id;

        Docsize _nd;  // number of words in document d

        double _ll;  // current log-likelihood
        double _best_ll;
        int    _best_iter;
        int    _iter;

        vector<double> _eta; // Smoother for document likelihood
        double _eta_sum;

        // Test for convergence
        unsigned _converged_iterations;
};

typedef std::pair<string, unsigned> word_score;
bool word_score_comp(const word_score& left, const word_score& right);

void init_random();

// Safely remove an element from a list
void safe_remove_crp(vector<CRP*>* v, const CRP*);

// These are adapted from Hal Daume's HBC:

// Logarithm of the gamma function.
double gammaln(double x);

// Log factorial
inline double factln(double x) { return gammaln(x+1); }

long double addLog(long double x, long double y);
void normalizeLog(vector<double>*x);
void normalizeLog(vector<pair<unsigned,double> >*x);

// Given a multinomial distribution of the form {label:prob}, return a label
// with that probability.
inline int sample_normalized_multinomial(vector<double>*d);
inline int sample_normalized_multinomial(vector<pair<unsigned,double> >*d);
int sample_unnormalized_log_multinomial(vector<double>*d);
int sample_unnormalized_log_multinomial(vector<pair<unsigned,double> >*d);
int SAFE_sample_unnormalized_log_multinomial(vector<double>*d);
int SAFE_sample_unnormalized_log_multinomial(vector<pair<unsigned,double> >*d);

unsigned sample_integer(unsigned range);
double sample_gaussian();
double sample_uniform();

string get_base_name(const string& s);
// filtering_ostream get_bz2_ostream(const string& filename);
bool is_bz2_file(const string& s);
#endif  // GIBBS_BASE_H_
