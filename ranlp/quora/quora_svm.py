__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import _pickle as p
import os

from models.svm import Model
from quora import Quora
from quora_bm25 import QuoraBM25
from quora_cosine import QuoraCosine, QuoraSoftCosine
from quora_translation import QuoraTranslations

from sklearn.preprocessing import MinMaxScaler

DATA_PATH='data'
FEATURES_PATH = '/roaming/tcastrof/quora/feature_final'
if not os.path.exists(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)

path = os.path.join(FEATURES_PATH, 'train')
if not os.path.exists(path):
    os.mkdir(path)

path = os.path.join(FEATURES_PATH, 'dev')
if not os.path.exists(path):
    os.mkdir(path)

class QuoraSVM(Quora):
    def __init__(self, model='svm', features='bm25,', comment_features='bm25,', stop=True, vector='word2vec', path=FEATURES_PATH, alpha=0.1, sigma=0.9, gridsearch='random'):
        Quora.__init__(self, stop=stop, vector=vector)
        self.path = path
        self.features = features.split(',')
        self.comment_features = comment_features.split(',')
        self.gridsearch = gridsearch
        self.svm = Model()

        self.model = model
        self.bm25 = QuoraBM25(stop=stop) if 'bm25' in self.features+self.comment_features else None
        self.cosine = QuoraCosine(stop=stop) if 'cosine' in self.features+self.comment_features else None
        self.softcosine = QuoraSoftCosine(stop=stop, vector=vector) if 'softcosine' in self.features+self.comment_features else None
        self.translation = QuoraTranslations(alpha=alpha, sigma=sigma, stop=stop, vector=self.vector) if 'translation' in self.features+self.comment_features else None

        self.train()


    def extract_features(self, pairdata, elmoidx, elmovec, fullelmoidx, fullelmovec):
        X, y = [], []
        feat = []

        for i, pair in enumerate(pairdata):
            try:
                percentage = round(float(i + 1) / len(pairdata), 2)
                print('Extracting features: ', percentage, i + 1, sep='\t', end = '\r')
                q1id = pair['qid1'] if 'qid1' in pair else str(i) + '1'
                q2id = pair['qid2'] if 'qid2' in pair else str(i) + '2'
                q1, q2 = pair['tokens_proc1'], pair['tokens_proc2']

                x = []

                if self.stop:
                    q1_emb = self.encode(q1id, q1, elmoidx, elmovec)
                else:
                    q1_emb = self.encode(q1id, q1, fullelmoidx, fullelmovec)

                # bm25
                if 'bm25' in self.features:
                    score = self.bm25.model(q1, q2id)
                    x.append(score)
                # softcosine
                if 'softcosine' in self.features:
                    if self.stop:
                        q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                    else:
                        q2_emb = self.encode(q2id, q2, fullelmoidx, fullelmovec)
                    score = self.softcosine.model(q1, q1_emb, q2, q2_emb)
                    x.append(score)
                # translation
                if 'translation' in self.features:
                    if self.stop:
                        q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                    else:
                        q2_emb = self.encode(q2id, q2, fullelmoidx, fullelmovec)
                    lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q2, q2_emb)
                    x.append(trlmprob)
                # cosine
                if 'cosine' in self.features:
                    score = self.cosine.model(q1, q2)
                    x.append(score)

                y_ = int(pair['is_duplicate'])
                feat.append((x, y_))
                X.append(x)
                y.append(y_)
            except:
                print('Error')
                print(pair)
        return feat, X, y


    def train(self):
        path = os.path.join(FEATURES_PATH, 'train', self.path)
        self.X, self.y = [], []
        if not os.path.exists(path):
            feat, self.X, self.y = self.extract_features(self.trainset, self.trainidx, self.trainelmo, self.fulltrainidx, self.fulltrainelmo)

            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))
            for row in feat:
                self.X.append(row[0])
                self.y.append(row[1])

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
        path = os.path.join(FEATURES_PATH, 'dev', self.path)
        if not os.path.exists(path):
            feat, X, y = self.extract_features(self.devset, self.devidx, self.develmo, self.fulldevidx, self.fulldevelmo)
            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))

        y_real, y_pred = [], []
        for i, pair in enumerate(feat):
            X = pair[0]
            X = self.scaler.transform([X])[0]
            score, pred_label = self.svm.score(X)
            y_pred.append(pred_label)

            real_label = pair[1]
            y_real.append(real_label)

        parameter_settings = self.svm.return_parameter_settings(clf=self.model)

        return y_real, y_pred, parameter_settings