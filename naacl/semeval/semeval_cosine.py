__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import copy
import os
from semeval import Semeval
from models.cosine import Cosine, SoftCosine

import paths

DATA_PATH=paths.DATA_PATH

class SemevalCosine(Semeval):
    def __init__(self, stop=True, lowercase=True, punctuation=True, proctrain=True, path=DATA_PATH):
        Semeval.__init__(self, stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain)
        self.path = path
        self.train()

    def train(self):
        self.model = Cosine()
        ftfidf, fdict = 'tfidf', 'dict'
        if self.proctrain:
            if self.lowercase:
                ftfidf += '.lower'
                fdict += '.lower'
            if self.stop:
                ftfidf += '.stop'
                fdict += '.stop'
            if self.punctuation:
                ftfidf += '.punct'
                fdict += '.punct'
        ftfidf += '.model'
        fdict += '.model'

        tfidf_path = os.path.join(self.path, ftfidf)
        dict_path = os.path.join(self.path, fdict)
        if not os.path.exists(tfidf_path):
            corpus = copy.copy(self.additional)
            for i, q1id in enumerate(self.trainset):
                query = self.trainset[q1id]
                if self.proctrain:
                    q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
                    q1 = self.remove_punctuation(q1) if self.punctuation else q1
                    q1 = self.remove_stopwords(q1) if self.stop else q1
                else:
                    q1 = query['tokens']
                corpus.append(q1)

                duplicates = query['duplicates']
                for duplicate in duplicates:
                    rel_question = duplicate['rel_question']
                    q2id = rel_question['id']
                    if self.proctrain:
                        q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                        q2 = self.remove_punctuation(q2) if self.punctuation else q2
                        q2 = self.remove_stopwords(q2) if self.stop else q2
                    else:
                        q2 = rel_question['tokens']
                    corpus.append(q2)

                    for comment in duplicate['rel_comments']:
                        q3id = comment['id']
                        if self.proctrain:
                            q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                            q3 = self.remove_punctuation(q3) if self.punctuation else q3
                            q3 = self.remove_stopwords(q3) if self.stop else q3
                        else:
                            q3 = comment['tokens']
                        if len(q3) == 0:
                            q3 = ['eos']
                        corpus.append(q3)

            self.model.init(corpus, dict_path, tfidf_path)
        else:
            self.model.load(dict_path, tfidf_path)

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
    def __init__(self, stop=True, lowercase=True, punctuation=True, proctrain=True, vector='word2vec', w2vdim=300, path=DATA_PATH):
        Semeval.__init__(self, stop=stop, lowercase=lowercase, punctuation=punctuation, vector=vector, proctrain=proctrain, w2vdim=w2vdim)
        self.path = path
        self.train()

    def train(self):
        self.model = SoftCosine()
        ftfidf, fdict = 'tfidf', 'dict'
        if self.proctrain:
            if self.lowercase:
                ftfidf += '.lower'
                fdict += '.lower'
            if self.stop:
                ftfidf += '.stop'
                fdict += '.stop'
            if self.punctuation:
                ftfidf += '.punct'
                fdict += '.punct'
        ftfidf += '.model'
        fdict += '.model'

        tfidf_path = os.path.join(self.path, ftfidf)
        dict_path = os.path.join(self.path, fdict)
        # if not os.path.exists(tfidf_path):
        corpus = copy.copy(self.additional)
        for i, q1id in enumerate(self.trainset):
            query = self.trainset[q1id]
            if self.proctrain:
                q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
                q1 = self.remove_punctuation(q1) if self.punctuation else q1
                q1 = self.remove_stopwords(q1) if self.stop else q1
            else:
                q1 = query['tokens']
            corpus.append(q1)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                if self.proctrain:
                    q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                    q2 = self.remove_punctuation(q2) if self.punctuation else q2
                    q2 = self.remove_stopwords(q2) if self.stop else q2
                else:
                    q2 = rel_question['tokens']
                corpus.append(q2)

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    if self.proctrain:
                        q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                        q3 = self.remove_punctuation(q3) if self.punctuation else q3
                        q3 = self.remove_stopwords(q3) if self.stop else q3
                    else:
                        q3 = comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']
                    corpus.append(q3)

        self.model.init(corpus, dict_path, tfidf_path)
        # else:
        #     self.model.load(dict_path, tfidf_path)

        del self.additional
        del self.trainset


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
                    q1emb = self.encode(q1id, q1, self.devidx, self.develmo)
                    q2emb = self.encode(q2id, q2, self.devidx, self.develmo)
                    score = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking


    def test(self, testdata, elmoidx, elmovec):
        self.testdata = testdata
        ranking = {}
        for j, q1id in enumerate(self.testdata):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testdata), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            for q2id in self.testdata[q1id]:
                pair = self.testdata[q1id][q2id]
                q1, q2 = pair['q1'], pair['q2']

                if self.vector == 'alignments':
                    score = self.model.score(q1, q2, self.alignments)
                else:
                    q1emb = self.encode(q1id, q1, elmoidx, elmovec)
                    q2emb = self.encode(q2id, q2, elmoidx, elmovec)
                    score = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking


    def pairs(self, testdata, elmoidx, elmovec):
        self.testdata = testdata

        ranking = {}
        for i, q1id in enumerate(self.testdata):
            ranking[q1id] = {}
            percentage = round(float(i + 1) / len(self.testdata), 2)
            print('Progress: ', percentage, i + 1, sep='\t', end = '\r')

            for q2id in self.testdata[q1id]:
                ranking[q1id][q2id] = {}
                q2 = self.testdata[q1id][q2id]['q2']

                for q3id in self.testdata[q1id]:
                    q3 = self.testdata[q1id][q3id]['q2']

                    if self.vector == 'alignments':
                        score = self.model.score(q2, q3, self.alignments)
                    else:
                        q2emb = self.encode(q2id, q2, elmoidx, elmovec)
                        q3emb = self.encode(q3id, q3, elmoidx, elmovec)
                        score = self.model(q2, q2emb, q3, q3emb)

                    ranking[q1id][q2id][q3id] = score

        return ranking