__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import _pickle as p
import os
import numpy as np

from multiprocessing import Pool
from models.treekernel import TreeKernel
from models.svm import Model
from semeval import Semeval

DATA_PATH='kernel'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
KERNEL_PATH = os.path.join(DATA_PATH, 'trainkernel.pickle')

class SemevalTreeKernel(Semeval):
    def __init__(self, alpha=0, decay=1, ignore_leaves=True, smoothed=True, vector='word2vec', tree='tree', kernel_path=KERNEL_PATH):
        Semeval.__init__(self, vector=vector, stop=False)
        self.path = kernel_path
        self.tree = tree
        self.memoization = {}
        self.svm = Model()
        self.flat_traindata()
        self.treekernel = TreeKernel(alpha, decay, ignore_leaves, smoothed)
        self.train()

        del self.additional


    def memoize(self, q1id, q1, q1_emb, q1_token2lemma, q2id, q2, q2_emb, q2_token2lemma):
        if q1id in self.memoization:
            if q2id in self.memoization[q1id]:
                return self.memoization[q1id][q2id]
        else:
            self.memoization[q1id] = {}

        if q2id in self.memoization:
            if q1id in self.memoization[q2id]:
                return self.memoization[q2id][q1id]
        else:
            self.memoization[q2id] = {}

        k = self.treekernel(q1, q1_emb, q1_token2lemma, q2, q2_emb, q2_token2lemma)
        self.memoization[q1id][q2id] = k
        self.memoization[q2id][q1id] = k

        return k

    def flat_traindata(self):
        self.flattraindata = []
        for q1id in self.traindata:
            for q2id in self.traindata[q1id]:
                self.flattraindata.append(self.traindata[q1id][q2id])

    def extract_features(self, procdata, elmoidx, elmovec):
        feat, X, y = {}, [], []

        for i, q1id in enumerate(procdata):
            percentage = round(float(i + 1) / len(self.traindata), 2)
            for q2id in procdata[q1id]:
                q_pair = procdata[q1id][q2id]

                x = []
                q1id = q_pair['q1_id']
                q1 = q_pair['q1_full']
                q1_tree = q_pair['q1_tree'] if self.tree == 'tree' else q_pair['subj_q1_tree']
                q1_emb = self.encode(q1id, q1, elmoidx, elmovec)
                q1_token2lemma = dict(zip(q1, q_pair['q1_lemmas']))
                kq1 = self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, q1id, q1_tree, q1_emb, q1_token2lemma)

                q2id = q_pair['q2_id']
                q2 = q_pair['q2_full']
                q2_tree = q_pair['q2_tree'] if self.tree == 'tree' else q_pair['subj_q2_tree']
                q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                q2_token2lemma = dict(zip(q2, q_pair['q2_lemmas']))
                kq2 = self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, q2id, q2_tree, q2_emb, q2_token2lemma)

                for j, c in enumerate(self.flattraindata):
                    print('Path: ', self.path,  'Progress: ', percentage, i + 1, j+1, sep=10 * ' ', end='\r')
                    c1id = c['q1_id']
                    c1 = c['q1_full']
                    c1_tree = c['q1_tree'] if self.tree == 'tree' else c['subj_q1_tree']
                    c1_emb = self.encode(c1id, c1, self.fulltrainidx, self.fulltrainelmo)
                    c1_token2lemma = dict(zip(c1, c['q1_lemmas']))
                    kc1 = self.memoize(c1id, c1_tree, c1_emb, c1_token2lemma, c1id, c1_tree, c1_emb, c1_token2lemma)

                    c2id = c['q2_id']
                    c2 = c['q2_full']
                    c2_tree = c['q2_tree'] if self.tree == 'tree' else c['subj_q2_tree']
                    c2_emb = self.encode(c2id, c2, self.fulltrainidx, self.fulltrainelmo)
                    c2_token2lemma = dict(zip(c2, c['q2_lemmas']))
                    kc2 = self.memoize(c2id, c2_tree, c2_emb, c2_token2lemma, c2id, c2_tree, c2_emb, c2_token2lemma)

                    kq1c1 = float(self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, c1id, c1_tree, c1_emb, c1_token2lemma)) / np.sqrt(kq1 * kc1)  # normalized
                    kq2c2 = float(self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, c2id, c2_tree, c2_emb, c2_token2lemma)) / np.sqrt(kq2 * kc2)  # normalized

                    k = kq1c1 + kq2c2
                    x.append(k)

                y_ = q_pair['label']
                feat[q1id] = { q2id : (x, y) }
                X.append(x)
                y.append(y_)
        return feat, X, y


    def train(self):
        path = os.path.join('kernel', 'train', self.path)
        if not os.path.exists(path):
            feat, self.X, self.y = self.extract_features(self.traindata, self.fulltrainidx, self.fulltrainelmo)

            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))
            for q1id in feat:
                for q2id in feat[q1id]:
                    self.X.append(feat[q1id][q2id][0])
                    self.y.append(feat[q1id][q2id][1])

        self.svm.train_svm(trainvectors=self.X, labels=self.y, c='search', kernel='precomputed', gamma='search', jobs=10)


    def validate(self):
        path = os.path.join('kernel', 'dev', self.path)
        if not os.path.exists(path):
            feat, X, y = self.extract_features(self.devdata, self.fulldevidx, self.fulldevelmo)
            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))

        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(feat):
            for q2id in feat[q1id]:
                X = feat[q1id][q2id][0]

                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = feat[q1id][q2id][1]
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        parameter_settings = self.svm.return_parameter_settings(clf='svm')

        return ranking, y_real, y_pred, parameter_settings

    def test(self, testdata, fullelmoidx, fullelmovec, test_='test2016'):
        if test_ == 'test2016':
            path = os.path.join('kernel', 'test2016', self.path)
        else:
            path = os.path.join('kernel', 'test2017', self.path)

        self.testdata = testdata
        if not os.path.exists(path):
            feat, X, y = self.extract_features(self.testdata, fullelmoidx, fullelmovec)
            p.dump(feat, open(path, 'wb'))
        else:
            feat = p.load(open(path, 'rb'))

        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(feat):
            for q2id in feat[q1id]:
                X = feat[q1id][q2id][0]

                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = feat[q1id][q2id][1]
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        parameter_settings = self.svm.return_parameter_settings(clf='svm')

        return ranking, y_real, y_pred, parameter_settings


def run(thread_id, smoothed, vector, path):
    print('Thread_id: ', thread_id)
    SemevalTreeKernel(smoothed=smoothed, vector=vector, kernel_path=path)

if __name__ == '__main__':
    path = os.path.join(DATA_PATH, 'kernel.word2vec+elmo.pickle')
    SemevalTreeKernel(smoothed=True, vector='word2vec+elmo', tree='subj_tree', kernel_path=path)

    path = os.path.join(DATA_PATH, 'kernel.word2vec.pickle')
    SemevalTreeKernel(smoothed=True, vector='word2vec', tree='subj_tree', kernel_path=path)

    path = os.path.join(DATA_PATH, 'kernel.pickle')
    SemevalTreeKernel(smoothed=False, vector='word2vec', tree='subj_tree', kernel_path=path)