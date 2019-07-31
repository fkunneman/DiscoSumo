__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import _pickle as p
import os
import numpy as np

from models.treekernel import TreeKernel
from models.svm import Model
from semeval import Semeval

from multiprocessing import Pool

DATA_PATH='kernel'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
    os.mkdir(os.path.join(DATA_PATH, 'train'))
    os.mkdir(os.path.join(DATA_PATH, 'dev'))
    os.mkdir(os.path.join(DATA_PATH, 'test2016'))
    os.mkdir(os.path.join(DATA_PATH, 'test2017'))
KERNEL_PATH = os.path.join(DATA_PATH, 'trainkernel.pickle')

class SemevalTreeKernel(Semeval):
    def __init__(self, alpha=0, decay=1, ignore_leaves=True, smoothed=True, vector='word2vec', w2vdim=300, lowercase=True, tree='tree', kernel_path=KERNEL_PATH):
        Semeval.__init__(self, vector=vector, stop=False, lowercase=lowercase, punctuation=False, w2vdim=w2vdim)
        self.path = kernel_path
        self.tree = tree
        self.memoization = {}
        self.svm = Model()
        self.flat_traindata()
        self.treekernel = TreeKernel(alpha=alpha, decay=decay, ignore_leaves=ignore_leaves, smoothed=smoothed, lowercase=lowercase)
        self.train()

        del self.additional


    def memoize(self, q1id, q1, q1_emb, q1_token2lemma, q2id, q2, q2_emb, q2_token2lemma, alignments):
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

        k = self.treekernel(q1, q1_emb, q1_token2lemma, q2, q2_emb, q2_token2lemma, alignments)
        self.memoization[q1id][q2id] = k
        self.memoization[q2id][q1id] = k

        return k


    def flat_traindata(self):
        self.flattraindata = []
        for q1id in self.traindata:
            for q2id in self.traindata[q1id]:
                self.flattraindata.append(self.traindata[q1id][q2id])


    def get_alignment(self, c1, c2):
        alignments = []
        for i, w in enumerate(c1):
            alignments_i = []

            for j, t in enumerate(c2):
                try:
                    w_t = self.alignments[t[0]][t][w[0]][w]
                except:
                    w_t = 0.0
                alignments_i.append(w_t)
            alignments.append(alignments_i)
        return alignments


    def extract_features(self, procdata, elmoidx, elmovec):
        feat, X, y = {}, [], []

        for i, q1id in enumerate(procdata):
            feat[q1id] = {}
            percentage = round(float(i + 1) / len(procdata), 2)
            for q2id in procdata[q1id]:
                q_pair = procdata[q1id][q2id]

                x = []
                q1id = q_pair['q1_id']
                q1 = q_pair['q1_full']
                q1_tree = q_pair['q1_tree'] if self.tree == 'tree' else q_pair['subj_q1_tree']
                q1_emb = self.encode(q1id, q1, elmoidx, elmovec)
                q1_token2lemma = dict(zip(q1, q_pair['q1_lemmas']))
                alignments = self.get_alignment(q1, q1) if self.vector == 'alignments' else []
                kq1 = self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, q1id, q1_tree, q1_emb, q1_token2lemma, alignments)

                q2id = q_pair['q2_id']
                q2 = q_pair['q2_full']
                q2_tree = q_pair['q2_tree'] if self.tree == 'tree' else q_pair['subj_q2_tree']
                q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                q2_token2lemma = dict(zip(q2, q_pair['q2_lemmas']))
                alignments = self.get_alignment(q2, q2) if self.vector == 'alignments' else []
                kq2 = self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, q2id, q2_tree, q2_emb, q2_token2lemma, alignments)

                if i % 10 == 0:
                    print('Path: ', self.path,  'Progress: ', percentage, i + 1, sep=10 * ' ', end='\r')
                for j, c in enumerate(self.flattraindata):
                    c1id = c['q1_id']
                    c1 = c['q1_full']
                    c1_tree = c['q1_tree'] if self.tree == 'tree' else c['subj_q1_tree']
                    c1_emb = self.encode(c1id, c1, self.trainidx, self.trainelmo)
                    c1_token2lemma = dict(zip(c1, c['q1_lemmas']))
                    alignments = self.get_alignment(c1, c1) if self.vector == 'alignments' else []
                    kc1 = self.memoize(c1id, c1_tree, c1_emb, c1_token2lemma, c1id, c1_tree, c1_emb, c1_token2lemma, alignments)

                    c2id = c['q2_id']
                    c2 = c['q2_full']
                    c2_tree = c['q2_tree'] if self.tree == 'tree' else c['subj_q2_tree']
                    c2_emb = self.encode(c2id, c2, self.trainidx, self.trainelmo)
                    c2_token2lemma = dict(zip(c2, c['q2_lemmas']))
                    alignments = self.get_alignment(c2, c2) if self.vector == 'alignments' else []
                    kc2 = self.memoize(c2id, c2_tree, c2_emb, c2_token2lemma, c2id, c2_tree, c2_emb, c2_token2lemma, alignments)

                    if kq1 == 0 or kc1 == 0:
                        kq1c1 = 0.0
                    else:
                        alignments = self.get_alignment(q1, c1) if self.vector == 'alignments' else []
                        kq1c1 = float(self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, c1id, c1_tree, c1_emb, c1_token2lemma, alignments)) / np.sqrt(kq1 * kc1)  # normalized

                    if kq2 == 0 or kc2 == 0:
                        kq2c2 = 0.0
                    else:
                        alignments = self.get_alignment(q2, c2) if self.vector == 'alignments' else []
                        kq2c2 = float(self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, c2id, c2_tree, c2_emb, c2_token2lemma, alignments)) / np.sqrt(kq2 * kc2)  # normalized

                    k = kq1c1 + kq2c2
                    x.append(k)

                y_ = q_pair['label']
                feat[q1id][q2id] = (x, y_)
                X.append(x)
                y.append(y_)
        return feat, X, y


    def train(self):
        path = os.path.join('kernel', 'train', self.path)
        self.X, self.y = [], []
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
        self.svm.train_svm(trainvectors=self.X, labels=self.y, c='search', kernel='precomputed', gamma='search', jobs=10)


    def validate(self):
        path = os.path.join('kernel', 'dev', self.path)
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

                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = feat[q1id][q2id][1]
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        parameter_settings = self.svm.return_parameter_settings(clf='svm')

        return ranking, y_real, y_pred, parameter_settings


    def test(self, testdata, elmoidx, elmovec, test_='test2016'):
        if test_ == 'test2016':
            path = os.path.join('kernel', 'test2016', self.path)
        elif test_ == 'train':
            path = os.path.join('kernel', 'train', self.path)
        elif test_ == 'dev':
            path = os.path.join('kernel', 'test2016', self.path)
        else:
            path = os.path.join('kernel', 'test2017', self.path)

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

                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = feat[q1id][q2id][1]
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        parameter_settings = self.svm.return_parameter_settings(clf='svm')

        return ranking, y_real, y_pred, parameter_settings

def run(smoothed, vector, tree, kernel_path, lowercase, w2vdim):
    s = SemevalTreeKernel(smoothed=smoothed, vector=vector, tree=tree, kernel_path=kernel_path, lowercase=lowercase, w2vdim=w2vdim)
    s.validate()
    s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

if __name__ == '__main__':
    pool = Pool(processes=4)

    processes = []
    # lower
    path = 'kernel.alignments.lower.pickle'
    processes.append(pool.apply_async(run, [True, 'alignments', 'subj_tree', path, True, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='alignments', tree='subj_tree', kernel_path=path, lowercase=True)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.fasttext+elmo.lower.pickle'
    processes.append(pool.apply_async(run, [True, 'fasttext+elmo', 'subj_tree', path, True, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='fasttext+elmo', tree='subj_tree', kernel_path=path, lowercase=True)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.fasttext.lower.pickle'
    processes.append(pool.apply_async(run, [True, 'fasttext', 'subj_tree', path, True, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='fasttext', tree='subj_tree', kernel_path=path, lowercase=True)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.word2vec+elmo.lower.pickle'
    processes.append(pool.apply_async(run, [True, 'word2vec+elmo', 'subj_tree', path, True, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='word2vec+elmo', tree='subj_tree', kernel_path=path, lowercase=True)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.word2vec.lower.pickle'
    processes.append(pool.apply_async(run, [True, 'word2vec', 'subj_tree', path, True, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='word2vec', tree='subj_tree', kernel_path=path, lowercase=True)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.lower.pickle'
    processes.append(pool.apply_async(run, [False, '', 'subj_tree', path, True, 300]))
    # s = SemevalTreeKernel(smoothed=False, vector='', tree='subj_tree', kernel_path=path, lowercase=True)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    # capital
    path = 'kernel.alignments.pickle'
    processes.append(pool.apply_async(run, [True, 'alignments', 'subj_tree', path, False, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='alignments', tree='subj_tree', kernel_path=path, lowercase=False)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.fasttext+elmo.pickle'
    processes.append(pool.apply_async(run, [True, 'fasttext+elmo', 'subj_tree', path, False, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='fasttext+elmo', tree='subj_tree', kernel_path=path, lowercase=False)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.fasttext.pickle'
    processes.append(pool.apply_async(run, [True, 'fasttext', 'subj_tree', path, False, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='fasttext', tree='subj_tree', kernel_path=path, lowercase=False)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.word2vec+elmo.pickle'
    processes.append(pool.apply_async(run, [True, 'word2vec+elmo', 'subj_tree', path, False, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='word2vec+elmo', tree='subj_tree', kernel_path=path, lowercase=False)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.word2vec.pickle'
    processes.append(pool.apply_async(run, [True, 'word2vec', 'subj_tree', path, False, 300]))
    # s = SemevalTreeKernel(smoothed=True, vector='word2vec', tree='subj_tree', kernel_path=path, lowercase=False)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    path = 'kernel.pickle'
    processes.append(pool.apply_async(run, [False, '', 'subj_tree', path, False, 300]))
    # s = SemevalTreeKernel(smoothed=False, vector='', tree='subj_tree', kernel_path=path, lowercase=False)
    # s.validate()
    # s.test(testdata=s.test2016data, elmoidx=s.test2016idx, elmovec=s.test2016elmo, test_='test2016')
    # s.test(testdata=s.test2017data, elmoidx=s.test2017idx, elmovec=s.test2017elmo, test_='test2017')

    for process in processes:
        doc = process.get()

    pool.close()
    pool.join()