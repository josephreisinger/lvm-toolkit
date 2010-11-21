import sys

MinimumDocCount = 2
MinimumContextCount = 3

context_doc_count = {}

for line in open(sys.argv[1]).readlines():
    for context in line.replace('\n','').split('\t')[1:]:
        c = context.split(':')[0]
        context_doc_count.setdefault(c,0)
        context_doc_count[c] += 1

    
for line in open(sys.argv[1]).readlines():
    contexts = [c for c in line.replace('\n','').split('\t')[1:] if context_doc_count[c.split(':')[0]] >= MinimumDocCount]
    if len(contexts) > MinimumContextCount:
        print '\t'.join([line.split('\t')[0]]+contexts)
