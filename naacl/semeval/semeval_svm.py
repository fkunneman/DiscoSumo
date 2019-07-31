__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import _pickle as p
import numpy as np
import os

from models.svm import Model
from semeval import Semeval
from semeval_bm25 import SemevalBM25
from semeval_cosine import SemevalCosine, SemevalSoftCosine
from semeval_translation import SemevalTranslation

from sklearn.preprocessing import MinMaxScaler

DATA_PATH='data'
FEATURES_PATH = 'feature'

class SemevalSVM(Semeval):
    def __init__(self, model='svm', features='bm25,', comment_features='bm25,', stop=True, vector='word2vec', lowercase=True, punctuation=True, proctrain=True, path=FEATURES_PATH, alpha=0.1, sigma=0.9, gridsearch='random'):
        Semeval.__init__(self, stop=stop, vector=vector, lowercase=lowercase, punctuation=punctuation)
        self.path = path
        self.features = features.split(',')
        self.comment_features = comment_features.split(',')
        self.gridsearch = gridsearch
        self.svm = Model()

        self.model = model
        self.bm25 = SemevalBM25(stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain) if 'bm25' in self.features+self.comment_features else None
        self.cosine = SemevalCosine(stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain) if 'cosine' in self.features+self.comment_features else None
        self.softcosine = SemevalSoftCosine(stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain, vector=vector) if 'softcosine' in self.features+self.comment_features else None
        self.translation = SemevalTranslation(alpha=alpha, sigma=sigma, stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain, vector=self.vector) if 'translation' in self.features+self.comment_features else None

        self.train()


    def extract_features(self, procdata, elmoidx, elmovec):
        X, y = [], []
        feat = {}
        for i, q1id in enumerate(procdata):
            feat[q1id] = {}
            percentage = round(float(i + 1) / len(procdata), 2)
            print('Extracting features: ', percentage, i + 1, sep='\t', end = '\r')
            for q2id in procdata[q1id]:
                query_question = procdata[q1id][q2id]
                q1, q2 = query_question['q1'], query_question['q2']
                x = []

                q1_emb = self.encode(q1id, q1, elmoidx, elmovec)

                # bm25
                if 'bm25' in self.features:
                    score = self.bm25.model(q1, q2id)
                    x.append(score)

                    for comment in query_question['comments']:
                        q3id = comment['id']
                        q3 = comment['tokens']

                        if len(q3) > 0:
                            score = self.bm25.model(q1, q3id)
                            x.append(score)
                        else:
                            x.append(0)

                # softcosine
                elif 'softcosine' in self.features:
                    if self.vector == 'alignments':
                        score = self.softcosine.model.score(q1, q2, self.alignments)
                    else:
                        q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                        score = self.softcosine.model(q1, q1_emb, q2, q2_emb)
                    x.append(score)

                    for comment in query_question['comments']:
                        q3id = comment['id']
                        q3 = comment['tokens']

                        if len(q3) > 0:
                            if self.vector == 'alignments':
                                score = self.softcosine.model.score(q1, q2, self.alignments)
                            else:
                                q3_emb = self.encode(q3id, q3, elmoidx, elmovec)
                                score = self.softcosine.model(q1, q1_emb, q3, q3_emb)
                            x.append(score)
                        else:
                            x.append(0)

                # translation
                elif 'translation' in self.features:
                    if self.vector == 'alignments':
                        lmprob, trmprob, trlmprob, proctime = self.translation.model.score(q1, q2)
                    else:
                        q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                        lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q2, q2_emb)
                    x.append(trlmprob)

                    for comment in query_question['comments']:
                        q3id = comment['id']
                        q3 = comment['tokens']

                        if len(q3) > 0:
                            if self.vector == 'alignments':
                                lmprob, trmprob, trlmprob, proctime = self.translation.model.score(q1, q3)
                            else:
                                q3_emb = self.encode(q3id, q3, elmoidx, elmovec)
                                lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q3, q3_emb)
                            x.append(trlmprob)
                        else:
                            x.append(0)

                # cosine
                elif 'cosine' in self.features:
                    score = self.cosine.model(q1, q2)
                    x.append(score)

                    for comment in query_question['comments']:
                        q3id = comment['id']
                        q3 = comment['tokens']

                        if len(q3) > 0:
                            score = self.cosine.model(q1, q3)
                            x.append(score)
                        else:
                            x.append(0)

                y_ = query_question['label']
                feat[q1id][q2id] = (x, y_)
                X.append(x)
                y.append(y_)
        return feat, X, y


    def train(self):
        self.X, self.y = [], []
        path = os.path.join('feature', 'train', self.path)
        if not os.path.exists(path):
            feat, self.X, self.y = self.extract_features(self.traindata, self.trainidx, self.trainelmo)

            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))
            for q1id in feat:
                for q2id in feat[q1id]:
                    self.X.append(feat[q1id][q2id][0])
                    self.y.append(feat[q1id][q2id][1])

        self.X = np.array(self.X)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

        if self.model == 'svm':
            self.svm.train_svm(
                trainvectors=self.X,
                labels=self.y,
                c='search',
                kernel='search',
                gamma='search',
                jobs=10,
                gridsearch=self.gridsearch
            )
        else:
            self.svm.train_regression(trainvectors=self.X, labels=self.y, c='search', penalty='search', tol='search', gridsearch=self.gridsearch, jobs=10)


    def validate(self):
        path = os.path.join('feature', 'dev', self.path)
        if not os.path.exists(path):
            feat, X, y = self.extract_features(self.devdata, self.devidx, self.develmo)
            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))

        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(feat):
            ranking[q1id] = []
            for q2id in feat[q1id]:
                X = feat[q1id][q2id][0]

                X = self.scaler.transform([X])[0]
                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = feat[q1id][q2id][1]
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        parameter_settings = self.svm.return_parameter_settings(clf=self.model)

        return ranking, y_real, y_pred, parameter_settings


    def test(self, testdata, elmoidx, elmovec, test_='test2016'):
        if test_ == 'test2016':
            path = os.path.join('feature', 'test2016', self.path)
        else:
            path = os.path.join('feature', 'test2017', self.path)

        self.testdata = testdata
        if not os.path.exists(path):
            feat, X, y = self.extract_features(self.testdata, elmoidx, elmovec)
            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))

        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(feat):
            ranking[q1id] = []
            for q2id in feat[q1id]:
                X = feat[q1id][q2id][0]

                X = self.scaler.transform([X])[0]
                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = feat[q1id][q2id][1]
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        parameter_settings = self.svm.return_parameter_settings(clf=self.model)

        return ranking, y_real, y_pred, parameter_settings