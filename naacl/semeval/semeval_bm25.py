__author__='thiagocastroferreira'

import sys
sys.path.append('../')

from semeval import Semeval
from models.bm25 import BM25

class SemevalBM25(Semeval):
    def __init__(self, stop=True, lowercase=True):
        Semeval.__init__(self, stop=stop, lowercase=lowercase)
        self.train()

    def train(self):
        corpus = dict([(i, q) for i, q in enumerate(self.additional)])

        for i, q1id in enumerate(self.trainset):
            query = self.trainset[q1id]
            q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
            corpus[q1id] = q1

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                corpus[q2id] = q2

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus[q3id] = q3

        for i, q1id in enumerate(self.devset):
            query = self.devset[q1id]
            q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
            corpus[q1id] = q1

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                corpus[q2id] = q2

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus[q3id] = q3

        for i, q1id in enumerate(self.testset2016):
            query = self.testset2016[q1id]
            q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
            corpus[q1id] = q1

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                corpus[q2id] = q2

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus[q3id] = q3

        for i, q1id in enumerate(self.testset2017):
            query = self.testset2017[q1id]
            q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
            corpus[q1id] = q1

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                corpus[q2id] = q2

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus[q3id] = q3
        self.model = BM25(corpus)

        del self.additional
        del self.trainset
        del self.traindata


    def validate(self):
        ranking = {}
        for i, q1id in enumerate(self.devdata):
            ranking[q1id] = []
            percentage = round(float(i + 1) / len(self.devdata), 2)
            print('Progress: ', percentage, i + 1, sep='\t', end = '\r')

            for q2id in self.devdata[q1id]:
                pair = self.devdata[q1id][q2id]
                q1 = pair['q1']
                q2score = self.model(q1, q2id)
                real_label = pair['label']
                ranking[q1id].append((real_label, q2score, q2id))
        return ranking


    def test(self, testdata):
        self.testdata = testdata
        ranking = {}
        for i, q1id in enumerate(self.testdata):
            ranking[q1id] = []
            percentage = round(float(i + 1) / len(self.testdata), 2)
            print('Progress: ', percentage, i + 1, sep='\t', end = '\r')

            for q2id in self.testdata[q1id]:
                pair = self.testdata[q1id][q2id]
                q1 = pair['q1']
                q2score = self.model(q1, q2id)
                real_label = pair['label']
                ranking[q1id].append((real_label, q2score, q2id))
        return ranking