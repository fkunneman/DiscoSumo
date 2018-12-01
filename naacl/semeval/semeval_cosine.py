__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import copy
import os
from semeval import Semeval
from models.cosine import Cosine, SoftCosine

DATA_PATH='data'

class SemevalCosine(Semeval):
    def __init__(self, stop=True, lowercase=True):
        Semeval.__init__(self, stop=stop, lowercase=lowercase)
        self.train()

    def train(self):
        self.model = Cosine()
        path = os.path.join(DATA_PATH,'tfidf.model')
        if not os.path.exists(path):
            corpus = copy.copy(self.additional)
            for i, q1id in enumerate(self.trainset):
                query = self.trainset[q1id]
                q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
                corpus.append(q1)

                duplicates = query['duplicates']
                for duplicate in duplicates:
                    rel_question = duplicate['rel_question']
                    q2id = rel_question['id']
                    q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                    corpus.append(q2)

                    for comment in duplicate['rel_comments']:
                        q3id = comment['id']
                        q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                        if len(q3) == 0:
                            q3 = ['eos']
                        corpus.append(q3)

            self.model.init(corpus, DATA_PATH)
        else:
            self.model.load(DATA_PATH)

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
                q2 = pair['q2']
                score = self.model(q1, q2)
                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
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
                q2 = pair['q2']
                score = self.model(q1, q2)
                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking


class SemevalSoftCosine(Semeval):
    def __init__(self, stop=True, lowercase=True, vector='word2vec'):
        Semeval.__init__(self, stop=stop, lowercase=lowercase, vector=vector)
        self.train()

    def train(self):
        self.model = Cosine()
        path = os.path.join(DATA_PATH,'tfidf.model')
        if not os.path.exists(path):
            corpus = copy.copy(self.additional)
            for i, q1id in enumerate(self.trainset):
                query = self.trainset[q1id]
                q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
                corpus.append(q1)

                duplicates = query['duplicates']
                for duplicate in duplicates:
                    rel_question = duplicate['rel_question']
                    q2id = rel_question['id']
                    q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                    corpus.append(q2)

                    for comment in duplicate['rel_comments']:
                        q3id = comment['id']
                        q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                        if len(q3) == 0:
                            q3 = ['eos']
                        corpus.append(q3)

            self.model.init(corpus, DATA_PATH)
        else:
            self.model.load(DATA_PATH)

        del self.additional
        del self.trainset
        del self.traindata


    def validate(self):
        ranking = {}
        for j, q1id in enumerate(self.devdata):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devdata), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            for q2id in self.devdata[q1id]:
                pair = self.devdata[q1id][q2id]
                q1, q2 = pair['q1'], pair['q2']

                if self.vector == 'alignments':
                    score = self.model.score(q1, q2, self.alignments)
                else:
                    if self.stop:
                        q1emb = self.encode(q1id, q1, self.devidx, self.develmo)
                        q2emb = self.encode(q2id, q2, self.devidx, self.develmo)
                    else:
                        q1emb = self.encode(q1id, q1, self.fulldevidx, self.fulldevelmo)
                        q2emb = self.encode(q2id, q2, self.fulldevidx, self.fulldevelmo)
                    score = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking


    def test(self, testdata, elmoidx, elmovec, fullelmoidx, fullelmovec):
        self.testdata = testdata
        ranking = {}
        for j, q1id in enumerate(self.testdata):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testdata), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            for q2id in self.testdata[q1id]:
                pair = self.devdata[q1id][q2id]
                q1, q2 = pair['q1'], pair['q2']

                if self.vector == 'alignments':
                    score = self.model.score(q1, q2, self.alignments)
                else:
                    if self.stop:
                        q1emb = self.encode(q1id, q1, elmoidx, elmovec)
                        q2emb = self.encode(q2id, q2, elmoidx, elmovec)
                    else:
                        q1emb = self.encode(q1id, q1, self.fulldevidx, self.fulldevelmo)
                        q2emb = self.encode(q2id, q2, fullelmoidx, fullelmovec)
                    score = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking