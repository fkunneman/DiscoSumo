__author='thiagocastroferreira'

import sys
sys.path.append('../')
import _pickle as p
import os

from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalSoftCosine
from semeval_kernel import SemevalTreeKernel

from models.svm import Model
from sklearn.preprocessing import MinMaxScaler

DATA_PATH='data'
ENSEMBLE_PATH='ensemble'
if not os.path.exists(ENSEMBLE_PATH):
    os.mkdir(ENSEMBLE_PATH)

class SemevalEnsemble:
    def __init__(self, stop=True, lowercase=True, punctuation=True, vector={}, scale=True, path='', alpha=0.8, sigma=0.2):
        self.stop = stop
        self.lowercase = lowercase
        self.punctuation = punctuation
        self.path = path
        self.scale = scale
        self.vector = vector
        self.alpha = alpha
        self.sigma = sigma

        self.ensemble = Model()
        self.train()


    def format(self, ranking):
        new_ranking = {}
        for q1id in ranking:
            new_ranking[q1id] = {}
            for question in ranking[q1id]:
                real_label, score, q2id = question
                new_ranking[q1id][q2id] = (score, real_label)
        return new_ranking


    def train(self):
        path = os.path.join('ensemble', 'bm25.' + self.path)
        if not os.path.exists(path):
            self.bm25 = SemevalBM25(stop=self.stop, lowercase=self.lowercase, punctuation=self.punctuation, proctrain=True)
            self.trainbm25 = self.format(self.bm25.test(self.bm25.traindata))
            self.devbm25 = self.format(self.bm25.validate())
            self.test2016bm25 = self.format(self.bm25.test(self.bm25.test2016data))
            self.test2017bm25 = self.format(self.bm25.test(self.bm25.test2017data))
            del self.bm25

            data = {'train': self.trainbm25, 'dev': self.devbm25, 'test2016': self.test2016bm25, 'test2017':self.test2017bm25}
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainbm25 = data['train']
            self.devbm25 = data['dev']
            self.test2016bm25 = data['test2016']
            self.test2017bm25 = data['test2017']

        vector = self.vector['translation']
        path = os.path.join('ensemble', 'translation.' + vector + '.' + self.path)
        if not os.path.exists(path):
            self.translation = SemevalTranslation(alpha=self.alpha, sigma=self.sigma, punctuation=self.punctuation, proctrain=True, vector=self.vector, stop=self.stop, lowercase=self.lowercase)
            self.traintranslation = self.format(self.translation.test(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo))
            self.devtranslation = self.format(self.translation.validate())
            self.test2016translation = self.format(self.translation.test(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo))
            self.test2017translation = self.format(self.translation.test(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo))
            del self.translation

            data = {'train': self.trainbm25, 'dev': self.devbm25, 'test2016': self.test2016bm25, 'test2017':self.test2017bm25}
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.traintranslation = data['train']
            self.devtranslation = data['dev']
            self.test2016translation = data['test2016']
            self.test2017translation = data['test2017']

        vector = self.vector['softcosine']
        path = os.path.join('ensemble', 'softcosine.' + vector + '.' + self.path)
        if not os.path.exists(path):
            self.softcosine = SemevalSoftCosine(stop=self.stop, vector=self.vector, lowercase=self.lowercase, punctuation=self.punctuation, proctrain=True)
            self.trainsoftcosine = self.format(self.softcosine.test(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo))
            self.devsoftcosine = self.format(self.softcosine.validate())
            self.test2016softcosine = self.format(self.softcosine.test(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo))
            self.test2017softcosine = self.format(self.softcosine.test(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo))
            del self.softcosine
        else:
            data = p.load(open(path, 'rb'))
            self.trainsoftcosine = data['train']
            self.devsoftcosine = data['dev']
            self.test2016softcosine = data['test2016']
            self.test2017softcosine = data['test2017']

        self.X, self.y = [], []

        for q1id in self.trainbm25:
            for q2id in self.trainbm25[q1id]:
                X = [self.trainbm25[q1id][q2id][0]]
                X.append(self.traintranslation[q1id][q2id][0])
                X.append(self.trainsoftcosine[q1id][q2id][0])
                self.X.append(X)
                self.y.append(self.trainbm25[q1id][q2id][1])

        if self.scale:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)
        self.ensemble.train_regression(trainvectors=self.X, labels=self.y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def test(self, set_='dev'):
        if set_ == 'dev':
            bm25 = self.devbm25
            translation = self.devtranslation
            softcosine = self.devsoftcosine
        elif set_ == 'train':
            bm25 = self.trainbm25
            translation = self.traintranslation
            softcosine = self.trainsoftcosine
        elif set_ == 'test2016':
            bm25 = self.test2016bm25
            translation = self.test2016translation
            softcosine = self.test2016softcosine
        else:
            bm25 = self.test2017bm25
            translation = self.test2017translation
            softcosine = self.test2017softcosine

        ranking = {}
        y_real, y_pred = [], []
        for q1id in bm25:
            ranking[q1id] = []
            for q2id in bm25[q1id]:
                X = []
                X.append(bm25[q1id][q2id][0])
                X.append(translation[q1id][q2id][0])
                X.append(softcosine[q1id][q2id][0])

                if self.scale:
                    X = self.scaler.transform([X])[0]
                score, pred_label = self.ensemble.score(X)
                y_pred.append(pred_label)

                real_label = 'true'
                y_real.append(real_label)
                ranking[q1id].append((pred_label, score, q2id))

        parameter_settings = self.ensemble.return_parameter_settings(clf='regression')
        return ranking, y_real, y_pred, parameter_settings


    def save(self, ranking, path, parameter_settings):
        with open(path, 'w') as f:
            f.write(parameter_settings)
            f.write('\n')
            for q1id in ranking:
                for row in ranking[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))