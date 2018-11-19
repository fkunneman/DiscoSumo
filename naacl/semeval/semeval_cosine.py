__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import copy
import os
from semeval import Semeval
from naacl.models.cosine import Cosine, SoftCosine

DATA_PATH='data'

class SemevalCosine(Semeval):
    def __init__(self):
        Semeval.__init__(self)
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
            self.model.init(corpus, path)
        else:
            self.model.load(path)


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


class SemevalSoftCosine(Semeval):
    def __init__(self, vector='word2vec'):
        Semeval.__init__(self)
        self.vector = vector
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
            self.model.init(corpus, path)
        else:
            self.model.load(path)


    def validate(self):
        ranking = {}
        for j, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']
            q1emb = self.encode(q1id, q1, self.devidx, self.develmo, self.vector)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_proc']

                if self.vector == 'alignments':
                    score = self.model.score(q1, q2, self.alignments)
                else:
                    q2emb = self.encode(q2id, q2, self.devidx, self.develmo, self.vector)
                    score = self.model(q1, q1emb, q2, q2emb)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, score, q2id))
        return ranking