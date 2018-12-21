__author__='thiagocastroferreira'

import sys
sys.path.append('../../')

import copy
import os
from models.cosine import SoftCosine
from semi import Semi

import paths

DATA_PATH=paths.DATA_PATH

class SemiSoftCosine(Semi):
    def __init__(self, stop=True, lowercase=True, punctuation=True, path=DATA_PATH):
        Semi.__init__(self, stop=stop, lowercase=lowercase, punctuation=punctuation)
        self.path = path
        self.train()

    def train(self):
        self.model = SoftCosine()
        ftfidf, fdict = 'tfidf', 'dict'
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
                q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
                q1 = self.remove_punctuation(q1) if self.punctuation else q1
                q1 = self.remove_stopwords(q1) if self.stop else q1
                corpus.append(q1)

                duplicates = query['duplicates']
                for duplicate in duplicates:
                    rel_question = duplicate['rel_question']
                    q2id = rel_question['id']
                    q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                    q2 = self.remove_punctuation(q2) if self.punctuation else q2
                    q2 = self.remove_stopwords(q2) if self.stop else q2
                    corpus.append(q2)

                    for comment in duplicate['rel_comments']:
                        q3id = comment['id']
                        q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                        q3 = self.remove_punctuation(q3) if self.punctuation else q3
                        q3 = self.remove_stopwords(q3) if self.stop else q3
                        if len(q3) == 0:
                            q3 = ['eos']
                        corpus.append(q3)

            self.model.init(corpus, dict_path, tfidf_path)
        else:
            self.model.load(dict_path, tfidf_path)

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

                q1emb = self.encode(q1)
                q2emb = self.encode(q2)
                score = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking


    def test(self, testdata):
        self.testdata = testdata
        ranking = {}
        for j, q1id in enumerate(self.testdata):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testdata), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            for q2id in self.testdata[q1id]:
                pair = self.testdata[q1id][q2id]
                q1, q2 = pair['q1'], pair['q2']

                q1emb = self.encode(q1)
                q2emb = self.encode(q2)
                score = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking