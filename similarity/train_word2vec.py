
import sys

import sentence_semantics

"""
Script to load sentences and train a w2v model
""" 

model_out = sys.argv[1]
corpora = sys.argv[2:]

sentences = []

# load in sentences
for corpus in corpora:
    print('loading in sentences for',corpus)
    with open(corpus,'r',encoding='utf-8') as corpus_in:
        sentences.extend([x.split() for x in corpus_in.read().strip().split('\n')])

print('done. loaded',len(sentences),'sentences')

print('now training w2v model...')
sensem = sentence_semantics.SentenceSemantics()
sensem.train_model(sentences)

print('done. saving model...')
sensem.save_model(model_out)
