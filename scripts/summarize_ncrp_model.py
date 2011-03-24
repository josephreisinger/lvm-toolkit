"""
Given a set of hlda samples, print out some statistics about them.
"""

import sys
import os
from model_file_utils import *
from operator import itemgetter
from collections import defaultdict

SampleBase = sys.argv[1]
MapOrBayes = sys.argv[2]
assert MapOrBayes in ['map', 'bayes']

ToShow = 5

# Load the model
# If we specify bayes, then average over the last ToKeep samples.
sys.stderr.write('loading model...\n')
(node_term_dist, term_node_dist, node_doc_dist, doc_node_dist, parents, alpha, eta, V, T) = load_all_ncrp_samples(SampleBase, MapOrBayes, ToKeep=10)

# Find the root node (hint: it doesn't have parents)
# root = [n for n in node_term_dist.iterkeys() if not parents[n]]
# assert len(root) == 1
# root = root[0]

# Print out a summary of the node-term distributions
for node, terms in node_term_dist.iteritems():
    print 'Node:', node, 'Parents:', parents[node]
    print '\n'.join(['\t%.1f: %s' % (f,w) for w,f in sorted(terms.iteritems(), key=itemgetter(1), reverse=True)[:ToShow]])
