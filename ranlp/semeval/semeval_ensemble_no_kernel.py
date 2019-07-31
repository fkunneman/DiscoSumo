__author__='thiagocastroferreira'

import paths
import sys
sys.path.append('../')
sys.path.append(paths.MAP_scripts)
import ev, metrics
import _pickle as p
import copy
import os

from operator import itemgetter
from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalSoftCosine

from models.svm import Model
from sklearn.preprocessing import MinMaxScaler

DEV_GOLD_PATH=paths.DEV_GOLD_PATH

DATA_PATH='data'
ENSEMBLE_PATH='ensemble'
if not os.path.exists(ENSEMBLE_PATH):
    os.mkdir(ENSEMBLE_PATH)

def prepare_gold(path):
    ir = ev.read_res_file_aid(path, 'trec')
    return ir


def evaluate(ranking, gold):
    for qid in gold:
        gold_sorted = sorted(gold[qid], key = itemgetter(2), reverse = True)
        pred_sorted = ranking[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(2), reverse = True)

        gold[qid], ranking[qid] = [], []
        for i, row in enumerate(gold_sorted):
            relevant, gold_score, aid = row
            gold[qid].append((relevant, gold_score, aid))

            pred_score = pred_sorted[i][1]
            ranking[qid].append((relevant, pred_score, aid))

    for qid in gold:
        # Sort by IR score.
        gold_sorted = sorted(gold[qid], key = itemgetter(1), reverse = True)

        # Sort by SVM prediction score.
        pred_sorted = ranking[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(1), reverse = True)

        gold[qid] = [rel for rel, score, aid in gold_sorted]
        ranking[qid] = [rel for rel, score, aid in pred_sorted]

    map_gold = metrics.map(gold, 10)
    map_pred = metrics.map(ranking, 10)
    return map_gold, map_pred

class SemevalEnsembleNoKernel:
    def __init__(self, stop={}, lowercase={}, punctuation={}, vector={}, scale=True, w2vdim=300, alpha=0.8, sigma=0.2):
        self.stop = stop
        self.lowercase = lowercase
        self.punctuation = punctuation
        self.scale = scale
        self.vector = vector
        self.alpha = alpha
        self.sigma = sigma
        self.w2vdim = w2vdim

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
        self.train_classifier()

        ranking = {}
        for q1id in self.devsoftcosine:
            ranking[q1id] = []
            for q2id in self.devsoftcosine[q1id]:
                X = []
                X.append(self.devbm25[q1id][q2id][0])
                X.append(self.devtranslation[q1id][q2id][0])
                X.append(self.devsoftcosine[q1id][q2id][0])

                if self.scale:
                    X = self.scaler.transform([X])[0]
                clfscore, pred_label = self.ensemble.score(X)

                ranking[q1id].append((pred_label, clfscore, q2id))

        map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))
        print('MAP baseline', map_baseline)
        print('MAP: ', map_model)
        print(10 * '-')


    def train_classifier(self):
        lowercase, stop, punctuation = self.lowercase['bm25'], self.stop['bm25'], self.punctuation['bm25']
        path = os.path.join('ensemble', 'bm25.lower_' + str(lowercase) + '.stop_' + str(stop) + '.punct_' + str(punctuation))
        if not os.path.exists(path):
            self.bm25 = SemevalBM25(stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=True)
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
        lowercase, stop, punctuation = self.lowercase['translation'], self.stop['translation'], self.punctuation['translation']
        path = os.path.join('ensemble', 'translation.lower_' + str(lowercase) + '.stop_' + str(stop) + '.punct_' + str(punctuation) + '.vector_' + str(vector) + '.vecdim_' + str(self.w2vdim))
        if not os.path.exists(path):
            self.translation = SemevalTranslation(alpha=self.alpha, sigma=self.sigma, punctuation=punctuation, proctrain=True, vector=vector, stop=stop, lowercase=lowercase, w2vdim=self.w2vdim)
            self.traintranslation = self.format(self.translation.test(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo))
            self.devtranslation = self.format(self.translation.validate())
            self.test2016translation = self.format(self.translation.test(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo))
            self.test2017translation = self.format(self.translation.test(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo))
            del self.translation

            data = {'train': self.traintranslation, 'dev': self.devtranslation, 'test2016': self.test2016translation, 'test2017':self.test2017translation}
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.traintranslation = data['train']
            self.devtranslation = data['dev']
            self.test2016translation = data['test2016']
            self.test2017translation = data['test2017']

        vector = self.vector['softcosine']
        lowercase, stop, punctuation = self.lowercase['softcosine'], self.stop['softcosine'], self.punctuation['softcosine']
        path = os.path.join('ensemble', 'softcosine.lower_' + str(lowercase) + '.stop_' + str(stop) + '.punct_' + str(punctuation) + '.vector_' + str(vector) + '.vecdim_' + str(self.w2vdim))
        if not os.path.exists(path):
            self.softcosine = SemevalSoftCosine(stop=stop, vector=vector, lowercase=lowercase, punctuation=punctuation, proctrain=True, w2vdim=self.w2vdim)
            self.trainsoftcosine = self.format(self.softcosine.test(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo))
            self.devsoftcosine = self.format(self.softcosine.validate())
            self.test2016softcosine = self.format(self.softcosine.test(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo))
            self.test2017softcosine = self.format(self.softcosine.test(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo))
            del self.softcosine

            data = {'train': self.trainsoftcosine, 'dev': self.devsoftcosine, 'test2016': self.test2016softcosine, 'test2017':self.test2017softcosine}
            p.dump(data, open(path, 'wb'))
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
                clfscore, pred_label = self.ensemble.score(X)
                y_pred.append(pred_label)

                real_label = 1 if bm25[q1id][q2id][1] == 'true' else 0
                y_real.append(real_label)

                ranking[q1id].append((pred_label, clfscore, q2id))

        parameter_settings = self.ensemble.return_parameter_settings(clf='regression')
        parameter_settings = parameter_settings + ',alpha=' + str(self.alpha) + ',sigma=' + str(self.sigma)
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