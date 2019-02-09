__author__='thiagocastroferreira'

from gensim.summarization import bm25

class BM25():
    def __init__(self, corpus={}):
        self.qid2idx = dict([(qid, i) for i, qid in enumerate(corpus.keys())])
        self.model = bm25.BM25(corpus.values())


    def __call__(self, q1, q2id):
        return self.model.get_score(q1, self.qid2idx[q2id])