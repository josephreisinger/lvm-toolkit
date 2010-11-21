import sys
from operator import itemgetter

NumberToShow = int(sys.argv[2])
ShowDocNames = int(sys.argv[3])

for (i, line) in enumerate(open(sys.argv[1]).readlines()[1:]):
    data = line.split('||')
    # x840e118   ||      ||  2137    ||  operations@@@198
    if ShowDocNames:
        word_counts = [x.split('@@@') for x in data[4].split('\t') if x]
        word_counts = [(x[1], int(x[2])) for x in word_counts]
    else:
        word_counts = [x.split('@@@') for x in data[3].split('\t') if x]
        word_counts = [(x[0], int(x[1])) for x in word_counts]

    print i, data[0]
    print '\n'.join(['\t[%d] %s' % (v,k) for (k,v) in sorted(word_counts,
        key=itemgetter(1), reverse=True)[:NumberToShow]])

