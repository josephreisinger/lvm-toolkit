"""
Utils for parsing wikipedia category graph
"""
import sys
import re
from string import lower
from bz2 import *
from collections import defaultdict
from operator import itemgetter
from glob import glob
from math import log, exp

def open_or_bz2(file):
    if file.endswith('.bz2'):
        import bz2
        return bz2.BZ2File(file)
    else:
        return open(file)

def log_add(x, y):
    if x == 0:
        return y
    if y == 0: 
        return x
    if x-y > 16:
        return x
    elif x > y:
        return x + log(1 + exp(y-x))
    elif y-x > 16:
        return y
    else:
        return y + log(1 + exp(x-y))

TOLERANCE = 0.000001

def read_docify(docify_file, parse_counts=True):
    if docify_file.endswith('.bz2'):
        f = open_or_bz2(docify_file)
    else:
        f = open(docify_file)
    for line in f:
        tokens = line.strip().split('\t')
        if parse_counts:
            doc_title, words = tokens[0], map(parse_count, tokens[1:])
        else:
            doc_title, words = tokens[0], tokens[1:]
        yield doc_title, words

def read_hlda_dictionary(dictionary_file):
    category_of = {}
    word_of = {}
    doc_of = {}
    doc_to_categories = {}
    unvisited_categories = []
    visited_categories = set()
    word_dictionary_section = False
    sys.stderr.write('Loading document-topic map...\n')
    doc_count = 0

    lines = open_or_bz2(dictionary_file)

    for line in lines:
        if line.strip():
            tokens = line.strip().split('\t')
            if word_dictionary_section:
                assert(len(tokens) == 2)
                key, word = tokens
                key = int(key, 0)
                assert(key not in word_of) 
                word_of[key] = word
            elif not unvisited_categories and not line.startswith('0x'):  # REALLY NASTY HACK
                document, doc_no, categories = tokens[0], tokens[1], tokens[2:]
                categories = [int(x.strip(),0) for x in categories]
                # print 'original', set(categories)
                # print 'adding', set(categories).difference(visited_categories)
                # TODO: fix this bug where there can be multiple _c entries for
                # the same topic
                unvisited_categories = [x for x in categories if not x in visited_categories]
                doc_to_categories[document] = categories
                doc_of[doc_count] = document
                doc_count += 1
            else:
                # print 'here bc not visited', unvisited_categories
                assert(len(tokens) == 2) # What
                key, category = tokens
                key = int(key, 0)
                # print '[%s]' % category
                # assert(category in unvisited_categories)
                assert(key not in category_of) 
                if not key in unvisited_categories:
                    sys.stderr.write('BUG: dup [%s] for %s\n' % (category,
                        document))
                else:    
                    unvisited_categories.remove(key)
                visited_categories.add(key)
                category_of[key] = category
        else:
            sys.stderr.write('Loading word dictionary...\n')
            word_dictionary_section = True

    return (category_of, word_of, doc_of, doc_to_categories)
    


def glob_samples(Moniker, MapOrBayes, ToKeep=20):
    if MapOrBayes == 'map':
        Header = Moniker + '*-best.hlda*'
        # Samples = [os.path.dirname(Header)+'/'+x for x in os.listdir(os.path.dirname(Header)) if x.startswith(os.path.basename(Header))] 
        Samples = glob(Header)
    else:
        Header = Moniker + '*-sample*'
        # Samples = [os.path.dirname(Header)+'/'+x for x in os.listdir(os.path.dirname(Header)) if x.startswith(os.path.basename(Header))] 
        Samples = glob(Header)
    sys.stderr.write('%s\n' % Header)
    sys.stderr.write('%d\n' % len(Samples))

    if not Samples:
        sys.stderr.write('FAIL No samples matched %s\n' % Header)
        sys.exit()

    if len(Samples) > ToKeep:
        Samples = Samples[::-1][:ToKeep]
        sys.stderr.write('keeping these %r\n' % Samples)

    return Samples

def load_all_ncrp_samples(Moniker, MapOrBayes, restrict_docs=False, ToKeep=20):
    """ Loads a set of samples from an ncrp and returns some sufficient
    statistics """
    node_term_dist = defaultdict(lambda: defaultdict(int))
    term_node_dist = defaultdict(lambda: defaultdict(int))
    node_doc_dist = defaultdict(lambda: defaultdict(int))
    doc_node_dist = defaultdict(lambda: defaultdict(int))
    prev = defaultdict(set)

    sys.stderr.write('Quantizing to TOLERANCE=%f\n' % TOLERANCE)

    alpha, eta, V, T = load_append_ncrp_samples(glob_samples(Moniker, MapOrBayes, ToKeep), 
                                                node_term_dist, 
                                                term_node_dist, 
                                                node_doc_dist, 
                                                doc_node_dist, 
                                                prev,
                                                restrict_docs=restrict_docs)

    return (node_term_dist, term_node_dist, node_doc_dist, doc_node_dist, prev,
            alpha, eta, V, T)

def ncrp_sample_iterator(sample):
    """
    Yields node data from an ncrp sample
    """
    visited = set()
    for (line_no, line) in enumerate(open_or_bz2(sample)):
        if line.startswith('ll ='):
            # ll = -434060662.375486 (-434060662.375486 at 123) -627048446 alpha = 0.001000 eta = 0.100000 L = 454
            # ll = -432994653.484200 (-432994653.484200 at 100) -1071382526 alpha = 0.001000 eta = 0.100000 L = 211
            m = re.search('alpha = (.*) eta = (.*) gamma .* L = (.*)', line)
            alpha = float(m.group(1))
            eta = float(m.group(2))
            L = int(m.group(3))
            sys.stderr.write('Got alpha = %f eta = %f L = %d\n' % (alpha,
                eta, L))
            continue

        line = line.replace('\n','')
        try:
            (node, _, m, raw_nw, raw_nd, tables) = line.split('||')
        except:
            sys.stderr.write('Excepted on %s\n' % sample)
            break

        # nodename is the actual memory address (uid of the node) nodelabel
        # is the tree label
        node = int(node.replace('\t',''), 0)

        assert node not in visited
        visited.add(node)

        parsed_nw = [x.rsplit('@@@') for x in raw_nw.split('\t') if x]
        parsed_nd = [x.rsplit('@@@') for x in raw_nd.split('\t') if x]

        nwsum = float(sum([int(c) for _,c in parsed_nw]))

        yield (node, parsed_nw, parsed_nd, nwsum, tables, alpha, eta, L)


def collect_term_term_count(Samples):
    """
    Collects the number of times two terms co-occur across all topics; this is
    normalized for frequency, so beware
    """

    # This version doesnt use the intermediate
    joint = defaultdict(float)
    marginal = defaultdict(float)
    for file_no, file in enumerate(Samples):
        for (node, nw, nd, nwsum, tables, alpha, eta, L) in ncrp_sample_iterator(file):
            for i, (word, count) in enumerate(nw):
                for k, (word2, count2) in enumerate(nw):
                    marginal[intern(word)] += 1.0
                    marginal[intern(word2)] += 1.0
                    if i < k:
                        (w1, w2) = sorted([word, word2])
                        joint[(intern(w1),intern(w2))] += 1.0

    return (joint, marginal)

def collect_term_pmi(Samples):
    """
    Computes the pmi from a joint and marginal distribution
    """
    joint, marginal = collect_term_term_count(Samples)
                        
    pmi_and_freq = defaultdict(float)
    for (w1,w2), freq in joint.iteritems():
        pmi_and_freq[(intern(w1),intern(w2))] = (log(freq) - log(marginal[w1]*marginal[w2]), freq)
    return pmi_and_freq



def load_append_ncrp_samples(Samples, node_term_dist, term_node_dist,
        node_doc_dist, doc_node_dist, prev, restrict_docs=set()):
    V = set()
    for file_no, file in enumerate(Samples):
        for (node, nw, nd, nwsum, tables, alpha, eta, L) in ncrp_sample_iterator(file):
            for (word, count) in nw:
                if float(count) / nwsum > TOLERANCE:
                    node_term_dist[node][intern(word)] += int(count)
                    term_node_dist[intern(word)][node] += int(count)
                V.add(intern(word))
            for (_, doc_name, count) in nd:
                node_doc_dist[node][intern(doc_name)] += int(count)
                doc_node_dist[intern(doc_name)][node] += int(count)

            # prev stores the DAG structure
            tables = [int(x,0) for x in tables.split('\t') if x != '']
            for t in tables:
                prev[t].add(node)

    return (alpha, eta, len(V), L)

def get_smoothed_terms_for(doc, doc_to_categories, node_doc_dist,
        node_term_dist, word_of, category_of, alpha=0,
        eta=0):
    pw = defaultdict(float)
    if not doc_to_categories.has_key(doc):
        sys.stderr.write('missing [%s]\n' % doc)
        return pw

    sys.stderr.write('Quantizing to TOLERANCE=%f\n' % TOLERANCE)

    T = len(doc_to_categories[doc])
    V = len(word_of)

    if alpha != None:
        sys.stderr.write('Smoothing with alpha=%f eta=%f\n' % (alpha,eta))

    dsum = float(sum([node_doc_dist[category_of[c]][doc] for c in
        doc_to_categories[doc]]))
    
    for raw_node in doc_to_categories[doc]:
        node = category_of[raw_node]

        d = node_doc_dist[node][doc]
        # print node, 'in ndd?', node_doc_dist.has_key(node)
        # print node, 'in ntd?', node_term_dist.has_key(node)
        sys.stderr.write('found %d/%d of [%s] in [%s]\n' % (d, dsum, doc, node))
        lpd = log(float(d)+alpha) - log(dsum+alpha*T)
        wsum = float(sum(node_term_dist[node].itervalues()))
        for word, w in node_term_dist[node].iteritems():
            lp = lpd + log(float(w)+eta) - log(wsum+eta*V)
            pw[intern(word)] = log_add(lp, pw[intern(word)])

    return pw

def HACK_ncrp_get_smoothed_terms_for(doc, doc_node_dist, node_doc_dist, node_term_dist, V, T, alpha=0, eta=0):
    """
    The bug here is that we can't get a list of all the possible nodes for doc;
    hence instead we have to just rely on where it is actually present
    (problematic)
    """
    pw = defaultdict(float)

    sys.stderr.write('USING HACK SMOOTHED TERMS\n')
    sys.stderr.write('Quantizing to TOLERANCE=%f\n' % TOLERANCE)

    if alpha != None:
        sys.stderr.write('Smoothing with alpha=%f eta=%f\n' % (alpha,eta))

    dsum = float(sum(doc_node_dist[doc].itervalues()))

    assert doc_node_dist.has_key(doc)

    for node, d in doc_node_dist[doc].iteritems():
        sys.stderr.write('found %d/%d of [%s] in [%s]\n' % (d, dsum, doc, node))
        lpd = log(float(d)+alpha) - log(dsum+alpha*T)
        wsum = float(sum(node_term_dist[node].itervalues()))
        for word, w in node_term_dist[node].iteritems():
            lp = lpd + log(float(w)+eta) - log(wsum+eta*V)
            pw[intern(word)] = log_add(lp, pw[intern(word)])

    return pw

def build_sort(dist, to_show=100):
    """
    Builds a sorted list summarizing the distribution
    """
    sorted_dist = []
    for concept in dist.keys():
        attribs = '\t%s' % '\n\t'.join(['%s %d' % (v, k) for (v, k) in
            sorted(dist[concept].items(), key=itemgetter(1),
                reverse=True)[:to_show]])
        count = sum(dist[concept].values())

        sorted_dist.append((count, concept, attribs))

    return sorted_dist

