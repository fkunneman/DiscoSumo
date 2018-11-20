__author__='thiagocastroferreira'

import sys
sys.path.append('../')

from semeval import Semeval
from models.bm25 import BM25

class SemevalBM25(Semeval):
    def __init__(self):
        Semeval.__init__(self)
        self.train()

    def train(self):
        corpus = dict([(i, q) for i, q in enumerate(self.additional)])

        for i, q1id in enumerate(self.trainset):
            query = self.trainset[q1id]
            q1 = query['tokens']
            corpus[q1id] = q1

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens']
                corpus[q2id] = q2

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus[q3id] = q3

        for i, q1id in enumerate(self.devset):
            query = self.devset[q1id]
            q1 = query['tokens']
            corpus[q1id] = q1

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens']
                corpus[q2id] = q2

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus[q3id] = q3
        self.model = BM25(corpus)

        del self.additional
        del self.trainset
        del self.traindata


    def validate(self):
        ranking = {}
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i + 1) / len(self.devset), 2)
            print('Progress: ', percentage, i + 1, sep='\t', end = '\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2score = self.model(q1, q2id)

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, q2score, q2id))
        return ranking