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

using namespace std;

// For vanilla LDA, we can save the state of the model directly in the raw
// docify, using an extra : for each term denoting the topic assignment. This
// allows us to do some fancy things like checkpoint and update documents on the
// fly.
DEFINE_int32(preassigned_topics,
              0,
              "Topics are preassigned in docify (vanilla only).");

// Streaming makes some major structural changes to the code, moving the main loop into load_data;
// This variable sets how many documents should be kept in memory at any one time
DEFINE_int32(streaming,
              0,
              "The number of documents to remember");

// Eta controls the amount of smoothing within the per-topic word distributions.
// Higher eta = more smoothing. Also used in the GEM sampler.
DEFINE_double(eta,
              0.1,
              "hyperparameter eta, controls the word smoothing");

// We can incorporate prior ranking information via eta. One way to do this is
// to assume that eta is proportional to some exponential of the average word
// rank, thus more highly ranked words have higher priors. eta_prior_scale
// represents the degree of confidence (how fast to decay) in each rank, higher
// meaning less informative prior.
DEFINE_double(eta_prior_exponential,
              1.0,
              "How should we decay the eta prior?");


// the number of gibbs iterations to run for.
DEFINE_int32(max_gibbs_iterations,
             99999,
             "Number of Gibbs sampling iterations to run.");


// One problem with Gibbs sampling is that nearby samples are highly
// correlated, throwing off the empirical distribution. In practice you need to
// wait some amount of time before reading each (independent) sample.
DEFINE_int32(sample_lag, 100, "how many Gibbs iterations to perform per sample (0 no samples)");

// The random seed
DEFINE_int32(random_seed, 101, "what seed value to use");

// Our tolerance for numerical (under)overflow.
DEFINE_double(epsilon_value, 1e-4, "tolerance for underflow");

// How many steps of not writing a best do we need before we declare
// convergence?
DEFINE_int32(convergence_interval,
             0,
             "How many steps should we wait before declaring convergence? 0 = off");

// Binarize the feature counts
DEFINE_bool(binarize,
            false,
            "Binarize the word counts.");

// The Mersenne Twister
dsfmt_t dsfmt;

void safe_remove_crp(vector<CRP*>* domain, const CRP* target) {
    vector<CRP*>::iterator p = find(domain->begin(), domain->end(), target);
    // must have existed
    CHECK(p != domain->end()) << "[" << target->label << "] didn't exist";
    domain->erase(p);
}

CRP::~CRP() {
    // TODO: want this check for the learned versions...
    // CHECK_LE(m, 1);  // don't delete without removing docs

    // LOG(INFO) << "about to remove from parents";
    if (!prev.empty()) {
        remove_from_parents();
    }

    // LOG(INFO) << "about to delete " << tables.size() << " tables";
    // remove all of our children nodes (note that this is recursive)
    for (int i = 0; i < tables.size(); i++) {
        delete tables[i];
    }
    // LOG(INFO) << "phew!!";
}

// Remove this node from the list of nodes stored at prev
void CRP::remove_from_parents() {
    // can only be called on an interior node (i.e. with a parent)
    CHECK_GT(prev.size(), 0);

    for (int i = 0; i < prev.size(); i++) {
        safe_remove_crp(&prev[i]->tables, this);
    }
}

bool GibbsSampler::sample_and_check_for_convergence() {
    if (_iter > 0) {  // Don't resample at first iteration, so we can accurately record the starting state
        resample_posterior();
        _converged_iterations += 1;
    }
    //if (i % FLAGS_sample_lag == 0) {  // calculate ll every 100
    _ll = compute_log_likelihood();
    //}
    CHECK_LE(_ll, 0) << "log likelihood cannot be positive!";

    if (_ll > _best_ll || _iter == 0) {
        _best_ll = _ll;
        _best_iter = _iter;
        _converged_iterations = 0;

        LOG(INFO) << "Resampling iter = " << _iter << " " << current_state() << " *";

        write_data("best");
    } else {
        LOG(INFO) << "Resampling iter = " << _iter << " " << current_state();
    }

    if (FLAGS_sample_lag > 0 && _iter % FLAGS_sample_lag == 0) {
        write_data(StringPrintf("sample-%05d", _iter));
    }

    _iter++;

    return FLAGS_convergence_interval > 0 && _converged_iterations >= FLAGS_convergence_interval;
}

void GibbsSampler::run() {
    _ll = compute_log_likelihood();

    while (_iter < FLAGS_max_gibbs_iterations) {

        if (sample_and_check_for_convergence()) {
            LOG(INFO) << "CONVERGED!";
            write_data("converged");
            break;
        }
    }
}

void GibbsSampler::process_document_line(const string& curr_line, unsigned line_no) {
    vector<string> words;
    vector<unsigned> encoded_words;
    vector<unsigned> topics;
    //CHECK_EQ(x, 0);

    SplitStringUsing(StringReplace(curr_line, "\n", "", true), "\t", &words);

    // V->insert(words.begin(), words.end());
    if (words.empty()) {
        LOG(WARNING) << "EMPTY LINE";
        return;
    }

    // the name of the document
    if (words.size() == 1) {
        LOG(WARNING) << "empty document " << words[0];
        return;
    }

    _document_name[line_no] = words[0];
    VLOG(1) << "found new document [" << words[0] << "] " << line_no;
    _document_id[words[0]] = line_no;
    _nd[line_no] = 0;

    for (int i = 1; i < words.size(); i++) {
        CHECK_STRNE(words[i].c_str(), "");
        // if (!(i == 0 && (HasPrefixString(words[i], "rpl_") ||
        //                 HasPrefixString(words[i], "RPL_")))) {
        vector<string> word_tokens;
        //VLOG(2) << words.at(i);
        SplitStringUsing(words.at(i), ":", &word_tokens);

        int topic;
        int freq;
        
        if (FLAGS_preassigned_topics == 1) {
            topic = atoi(word_tokens.back().c_str());
            word_tokens.pop_back();
        }
        
        freq = atoi(word_tokens.back().c_str());
        word_tokens.pop_back();

        if (FLAGS_preassigned_topics == 1) {
            CHECK_EQ(freq, 1);  // Each term gets a unique assignment
        }


        string word = JoinStrings(word_tokens, ":");

        VLOG(1) << word << " " << freq;
        if (_word_name_to_id.find(word) == _word_name_to_id.end()) {
            _word_name_to_id[word] = _unique_word_count;
            _word_id_to_name[_unique_word_count] = word;

            _unique_word_count += 1;

            _eta.push_back(FLAGS_eta);
            _eta_sum += FLAGS_eta;
        }
        _V[_word_name_to_id[word]] += freq;
        if (FLAGS_binarize) {
            freq = 1;
        }
        for (int f = 0; f < freq; f++) {
            encoded_words.push_back(_word_name_to_id[word]);
            topics.push_back(topic);
        }
        _total_word_count += freq;
        _nd[line_no] += freq;
    }
    _D[line_no] = encoded_words;
    _initial_topic_assignment[line_no] = topics;

    _lD = _D.size();
    _lV = _V.size();

    // Make sure eta is in a reasonable range
    CHECK_LT(_eta_sum, 1000000);

    if (FLAGS_streaming > 0) {
        streaming_step(line_no);
    }

}

void GibbsSampler::streaming_step(unsigned new_d) {
    allocate_document(new_d);
    
    if (_D.size() > FLAGS_streaming) {
        deallocate_document();
        _lD = _D.size();
        _lV = _V.size();
        sample_and_check_for_convergence();
    }
}

void GibbsSampler::load_data(const string& filename) {
    _D.clear();
    _V.clear();
    _word_id_to_name.clear();

    _initial_topic_assignment.clear();
    

    LOG(INFO) << "loading data from [" << filename << "]";

    ifstream input_file(filename.c_str(), ios_base::in | ios_base::binary);

    CHECK(input_file.is_open());

    string curr_line;
    unsigned line_no = 0;
    while (true) {
        if (input_file.eof()) {
            break;
        }
        getline(input_file, curr_line);
        process_document_line(curr_line, line_no);
        line_no += 1;
    }

    // Allocate documents
    if (FLAGS_streaming == 0) {
        batch_allocation();

        LOG(INFO) << "Loaded " << _lD << " documents with "
            << _total_word_count << " words (" << _V.size() << " unique) from "
            << filename;
    }
}

// Machinering for printing out the tops of multinomials
typedef std::pair<string, unsigned> word_score;
bool word_score_comp(const word_score& left, const word_score& right)
{
    return left.second > right.second;
}

string GibbsSampler::show_chopped_sorted_nw(const WordToCountMap& nw) {
    vector<word_score> sorted;
    for (WordToCountMap::const_iterator nw_itr = nw.begin();
            nw_itr != nw.end();
            nw_itr++) {
        unsigned w = nw_itr->first;
        unsigned c = nw_itr->second;
        if (c > 0) {
            sorted.push_back(make_pair(_word_id_to_name[w], c));
        }
    }

    sort(sorted.begin(), sorted.end(), word_score_comp);

    // Finally print out the summary
    string buffer = "";
    for (int k = 0; k < min((int)sorted.size(), 10); k++) {
        buffer += StringPrintf("%s %d ", sorted[k].first.c_str(), sorted[k].second);
    }

    return buffer;
}


void init_random() {
#ifdef USE_MT_RANDOM
    dsfmt_init_gen_rand(&dsfmt, FLAGS_random_seed);
#else
    srand(FLAGS_random_seed);
#endif

}

// Logarithm of the gamma function.
//
// References:
//
// 1) W. J. Cody and K. E. Hillstrom, 'Chebyshev Approximations for
// the Natural Logarithm of the Gamma Function,' Math. Comp. 21,
// 1967, pp. 198-203.
//
// 2) K. E. Hillstrom, ANL/AMD Program ANLC366S, DGAMMA/DLGAMA, May,
// 1969.
//
// 3) Hart, Et. Al., Computer Approximations, Wiley and sons, New
// York, 1968.
//
// From matlab/gammaln.m
double gammaln(double x) {
    double result, y, xnum, xden;
    int i;
    static double d1 = -5.772156649015328605195174e-1;
    static double p1[] = {
        4.945235359296727046734888e0, 2.018112620856775083915565e2,
        2.290838373831346393026739e3, 1.131967205903380828685045e4,
        2.855724635671635335736389e4, 3.848496228443793359990269e4,
        2.637748787624195437963534e4, 7.225813979700288197698961e3
    };
    static double q1[] = {
        6.748212550303777196073036e1, 1.113332393857199323513008e3,
        7.738757056935398733233834e3, 2.763987074403340708898585e4,
        5.499310206226157329794414e4, 6.161122180066002127833352e4,
        3.635127591501940507276287e4, 8.785536302431013170870835e3
    };
    static double d2 = 4.227843350984671393993777e-1;
    static double p2[] = {
        4.974607845568932035012064e0, 5.424138599891070494101986e2,
        1.550693864978364947665077e4, 1.847932904445632425417223e5,
        1.088204769468828767498470e6, 3.338152967987029735917223e6,
        5.106661678927352456275255e6, 3.074109054850539556250927e6
    };
    static double q2[] = {
        1.830328399370592604055942e2, 7.765049321445005871323047e3,
        1.331903827966074194402448e5, 1.136705821321969608938755e6,
        5.267964117437946917577538e6, 1.346701454311101692290052e7,
        1.782736530353274213975932e7, 9.533095591844353613395747e6
    };
    static double d4 = 1.791759469228055000094023e0;
    static double p4[] = {
        1.474502166059939948905062e4, 2.426813369486704502836312e6,
        1.214755574045093227939592e8, 2.663432449630976949898078e9,
        2.940378956634553899906876e10, 1.702665737765398868392998e11,
        4.926125793377430887588120e11, 5.606251856223951465078242e11
    };
    static double q4[] = {
        2.690530175870899333379843e3, 6.393885654300092398984238e5,
        4.135599930241388052042842e7, 1.120872109616147941376570e9,
        1.488613728678813811542398e10, 1.016803586272438228077304e11,
        3.417476345507377132798597e11, 4.463158187419713286462081e11
    };
    static double c[] = {
        -1.910444077728e-03, 8.4171387781295e-04,
        -5.952379913043012e-04, 7.93650793500350248e-04,
        -2.777777777777681622553e-03, 8.333333333333333331554247e-02,
        5.7083835261e-03
    };
    static double a = 0.6796875;

    if ((x <= 0.5) || ((x > a) && (x <= 1.5))) {
        if (x <= 0.5) {
            result = -log(x);
            /*  Test whether X < machine epsilon. */
            if (x+1 == 1) {
                return result;
            }
        } else {
            result = 0;
            x = (x - 0.5) - 0.5;
        }
        xnum = 0;
        xden = 1;
        for (i = 0; i < 8; i++) {
            xnum = xnum * x + p1[i];
            xden = xden * x + q1[i];
        }
        result += x * (d1 + x * (xnum / xden));
    } else if ((x <= a) || ((x > 1.5) && (x <= 4))) {
        if (x <= a) {
            result = -log(x);
            x = (x - 0.5) - 0.5;
        } else {
            result = 0;
            x -= 2;
        }
        xnum = 0;
        xden = 1;
        for (i = 0; i < 8 ;i++) {
            xnum = xnum * x + p2[i];
            xden = xden * x + q2[i];
        }
        result += x * (d2 + x * (xnum / xden));
    } else if (x <= 12) {
        x -= 4;
        xnum = 0;
        xden = -1;
        for (i = 0; i < 8; i++) {
            xnum = xnum * x + p4[i];
            xden = xden * x + q4[i];
        }
        result = d4 + x*(xnum/xden);
    } else {
        //  X > 12
        y = log(x);
        result = x * (y - 1) - y * 0.5 + .9189385332046727417803297;
        x = 1/x;
        y = x*x;
        xnum = c[6];
        for (i = 0; i < 6; i++) {
            xnum = xnum * y + c[i];
        }
        xnum *= x;
        result += xnum;
    }
    return result;
}


long double addLog(long double x, long double y) {
    if (x == 0) {
        return y;
    }
    if (y == 0) {
        return x;
    }

    if (x-y > 16) {
        return x;
    } else if (x > y) {
        return x + log(1 + exp(y-x));
    } else if (y-x > 16) {
        return y;
    } else {
        return y + log(1 + exp(x-y));
    }
}

void normalizeLog(vector<double>*x) {
    long double s;
    int i;
    s = 0;

    long double normalized_sum = 0;

    for (i = 0; i < x->size(); i++) {
        s = addLog(s, x->at(i));
    }
    for (i = 0; i < x->size(); i++) {
        (*x)[i] = exp(x->at(i) - s);
        normalized_sum += (*x)[i];
    }

    // CHECK(MathUtil::NearByMargin(normalized_sum,1.0));
    // LOG(INFO) << "normalized sum " << normalized_sum;
    CHECK_GT(normalized_sum, 0);  // for nan
    CHECK_LT(fabs(normalized_sum - 1.0), FLAGS_epsilon_value);
}
void normalizeLog(vector<pair<unsigned,double> >*x) {
    long double s;
    int i;
    s = 0;

    long double normalized_sum = 0;

    for (i = 0; i < x->size(); i++) {
        s = addLog(s, x->at(i).second);
    }
    for (i = 0; i < x->size(); i++) {
        (*x)[i] = pair<unsigned,double>(x->at(i).first, exp(x->at(i).second - s));
        normalized_sum += x->at(i).second;
    }

    // CHECK(MathUtil::NearByMargin(normalized_sum,1.0));
    // LOG(INFO) << "normalized sum " << normalized_sum;
    CHECK_GT(normalized_sum, 0);  // for nan
    CHECK_LT(fabs(normalized_sum - 1.0), FLAGS_epsilon_value);
}

double sample_uniform() {
#ifdef USE_MT_RANDOM
    return dsfmt_genrand_close_open(&dsfmt);
#else
    return random() / (double)RAND_MAX;
#endif

}

// Given a multinomial distribution of the form {label:prob}, return a label
// with that probability.
inline int sample_normalized_multinomial(vector<double>*d) {
    double cut = sample_uniform();
    CHECK_LE(cut, 1.0);
    CHECK_GE(cut, 0.0);

    for (int i = 0; i < d->size(); i++) {
        cut -= d->at(i);

        if (cut < 0) {
            return i;
        }
    }

    CHECK(false) << "improperly normalized distribution " << cut;
    return 0;
}
// Given a multinomial distribution of the form {label:prob}, return a label
// with that probability.
inline int sample_normalized_multinomial(vector<pair<unsigned,double> >*d) {
    double cut = sample_uniform();
    CHECK_LE(cut, 1.0);
    CHECK_GE(cut, 0.0);

    for (int i = 0; i < d->size(); i++) {
        cut -= d->at(i).second;

        if (cut < 0) {
            return d->at(i).first;
        }
    }

    CHECK(false) << "improperly normalized distribution " << cut;
    return -1;
}


// Assume that the data coming in are log probs and that they need to be
// appropriately normalized.
// XXX: sample_unnormalized_log_multinomial changes d into normal p space
int sample_unnormalized_log_multinomial(vector<double>*d) {
    double cut = sample_uniform();
    CHECK_LE(cut, 1.0);
    CHECK_GE(cut, 0.0);

    int i;
    long double s = 0;
    for (i = 0; i < d->size(); i++) {
        s = addLog(s, d->at(i));
    }
    for (i = 0; i < d->size(); i++) {
        cut -= exp(d->at(i) - s);

        if (cut < 0) {
            return i;
        }
    }

    CHECK(false) << "improperly normalized distribution " << cut;
    return 0;
}
int sample_unnormalized_log_multinomial(vector<pair<unsigned,double> >*d) {
    double cut = sample_uniform();
    CHECK_LE(cut, 1.0);
    CHECK_GE(cut, 0.0);

    int i;
    long double s = 0;

    for (i = 0; i < d->size(); i++) {
        s = addLog(s, d->at(i).second);
    }
    for (i = 0; i < d->size(); i++) {
        cut -= exp(d->at(i).second - s);

        if (cut < 0) {
            return d->at(i).first;
        }
    }

    CHECK(false) << "improperly normalized distribution " << cut;
    return -1;
}

int SAFE_sample_unnormalized_log_multinomial(vector<double>*d) {
    normalizeLog(d);
    return sample_normalized_multinomial(d);
}
int SAFE_sample_unnormalized_log_multinomial(vector<pair<unsigned,double> >*d) {
    normalizeLog(d);
    return sample_normalized_multinomial(d);
}

unsigned sample_integer(unsigned range) {
    return (unsigned)(sample_uniform() * range);
}

double sample_gaussian() {
    double x1, x2, w, y1;

    static bool returned = false;
    static double y2 = 0.0;

    if (returned) {
        returned = false;
        do {
            x1 = 2.0 * sample_uniform() - 1.0;
            x2 = 2.0 * sample_uniform() - 1.0;
            w = x1 * x1 + x2 * x2;
        } while ( w >= 1.0 );

        w = sqrt((-2.0 * log(w)) / w);
        y1 = x1 * w;
        y2 = x2 * w;
        return y1;
    } else {
        returned = true;
        return y2;
    }
}


// Returns the file part of the path s
string get_base_name(const string& s) {
    vector<string> tokens;
    SplitStringUsing(s, "/", &tokens);
    return tokens.back();
}

// Test whether this string ends in bz2
bool is_bz2_file(const string& s) {
    vector<string> tokens;
    SplitStringUsing(s, ".", &tokens);

    return tokens.back() == "bz2";
}
