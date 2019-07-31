__author__='thiagocastroferreira'

import sys
sys.path.append('../')

from quora import Quora
from models.bm25 import BM25

class QuoraBM25(Quora):
    def __init__(self, stop=True):
        Quora.__init__(self, stop=stop)
        self.train()

    def train(self):
        corpus = {}

        for i, pair in enumerate(self.trainset):
            q1id = pair['qid1']
            q1 = pair['tokens1']
            corpus[q1id] = q1

            q2id = pair['qid2']
            q2 = pair['tokens2']
            corpus[q2id] = q2

        for i, pair in enumerate(self.devset):
            q1id = pair['qid1']
            q1 = pair['tokens1']
            corpus[q1id] = q1

            q2id = pair['qid2']
            q2 = pair['tokens2']
            corpus[q2id] = q2

        # for i, pair in enumerate(self.testset):
        #     q1id = pair['qid1']
        #     q1 = pair['tokens1']
        #     corpus[q1id] = q1
        #
        #     q2id = pair['qid2']
        #     q2 = pair['tokens2']
        #     corpus[q2id] = q2

        self.model = BM25(corpus)

        del self.trainset