__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import copy
import os
from semeval import Semeval
from models.cosine import Cosine, SoftCosine

DATA_PATH='data'

class SemevalCosine(Semeval):
    def __init__(self, stop=True):
        Semeval.__init__(self, stop=stop)
        self.train()

    def train(self):
        self.model = Cosine()
        path = os.path.join(DATA_PATH,'tfidf.model')
        if not os.path.exists(path):
            corpus = copy.copy(self.additional)
            for qid in self.trainset:
                q1 = self.trainset[qid]['tokens']

                duplicates = self.trainset[qid]['duplicates']
                for duplicate in duplicates:
                    q2 = duplicate['rel_question']['tokens']
                    corpus.append(q2)

                    rel_comments = duplicate['rel_comments']
                    for rel_comment in rel_comments:
                        q3 = rel_comment['tokens']
                        corpus.append(q3)

                corpus.append(q1)
            self.model.init(corpus, DATA_PATH)
        else:
            self.model.load(DATA_PATH)

        del self.additional
        del self.trainset
        del self.traindata


    def validate(self):
        ranking = {}
        for j, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_proc']

                score = self.model(q1, q2)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, score, q2id))
        return ranking

    def test(self, testset):
        self.testset = testset
        ranking = {}
        for j, q1id in enumerate(self.testset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_proc']

                score = self.model(q1, q2)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, score, q2id))
        return ranking


class SemevalSoftCosine(Semeval):
    def __init__(self, stop=True, vector='word2vec'):
        Semeval.__init__(self, stop=stop, vector=vector)
        self.train()

    def train(self):
        self.model = SoftCosine()
        path = os.path.join(DATA_PATH,'tfidf.model')
        if not os.path.exists(path):
            corpus = copy.copy(self.additional)
            for qid in self.trainset:
                q1 = self.trainset[qid]['tokens']

                duplicates = self.trainset[qid]['duplicates']
                for duplicate in duplicates:
                    q2 = duplicate['rel_question']['tokens']
                    corpus.append(q2)

                    rel_comments = duplicate['rel_comments']
                    for rel_comment in rel_comments:
                        q3 = rel_comment['tokens']
                        corpus.append(q3)

                corpus.append(q1)
            self.model.init(corpus, DATA_PATH)
        else:
            self.model.load(DATA_PATH)

        del self.additional
        del self.trainset
        del self.traindata


    def validate(self):
        ranking = {}
        for j, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc'] if self.stop else query['tokens']
            if self.stop:
                q1emb = self.encode(q1id, q1, self.devidx, self.develmo)
            else:
                q1emb = self.encode(q1id, q1, self.fulldevidx, self.fulldevelmo)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_proc'] if self.stop else rel_question['tokens']

                if self.vector == 'alignments':
                    score = self.model.score(q1, q2, self.alignments)
                else:
                    if self.stop:
                        q2emb = self.encode(q2id, q2, self.devidx, self.develmo)
                    else:
                        q2emb = self.encode(q2id, q2, self.fulldevidx, self.fulldevelmo)
                    score = self.model(q1, q1emb, q2, q2emb)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, score, q2id))
        return ranking

    def test(self, testset, elmoidx, elmovec, fullelmoidx, fullelmovec):
        self.testset = testset
        ranking = {}
        for j, q1id in enumerate(self.testset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.testset[q1id]
            q1 = query['tokens_proc'] if self.stop else query['tokens']
            if self.stop:
                q1emb = self.encode(q1id, q1, elmoidx, elmovec)
            else:
                q1emb = self.encode(q1id, q1, fullelmoidx, fullelmovec)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_proc'] if self.stop else rel_question['tokens']

                if self.vector == 'alignments':
                    score = self.model.score(q1, q2, self.alignments)
                else:
                    if self.stop:
                        q2emb = self.encode(q2id, q2, elmoidx, elmovec)
                    else:
                        q2emb = self.encode(q2id, q2, fullelmoidx, fullelmovec)
                    score = self.model(q1, q1emb, q2, q2emb)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, score, q2id))
        return ranking