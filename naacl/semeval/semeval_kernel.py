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


    def train(self):
        if not os.path.exists(self.path):
            self.X, self.y = [], []
            for i, q_pair in enumerate(self.traindata):
                percentage = round(float(i + 1) / len(self.traindata), 2)
                x = []
                q1id = q_pair['q1_id']
                q1 = q_pair['q1_full']
                q1_tree = q_pair['q1_tree'] if self.tree == 'tree' else q_pair['subj_q1_tree']
                q1_emb = self.encode(q1id, q1, self.fulltrainidx, self.fulltrainelmo)
                q1_token2lemma = dict(zip(q1, q_pair['q1_lemmas']))
                kq1 = self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, q1id, q1_tree, q1_emb, q1_token2lemma)

                q2id = q_pair['q2_id']
                q2 = q_pair['q2_full']
                q2_tree = q_pair['q2_tree'] if self.tree == 'tree' else q_pair['subj_q2_tree']
                q2_emb = self.encode(q2id, q2, self.fulltrainidx, self.fulltrainelmo)
                q2_token2lemma = dict(zip(q2, q_pair['q2_lemmas']))
                kq2 = self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, q2id, q2_tree, q2_emb, q2_token2lemma)

                for j, c in enumerate(self.traindata):
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
                self.X.append(x)
                self.y.append(q_pair['label'])

            p.dump(list(zip(self.X, self.y)), open(self.path, 'wb'))
            self.X = np.array(self.X)
        else:
            f = p.load(open(self.path, 'rb'))
            self.X = np.array([x[0] for x in f])
            self.y = list(map(lambda x: x[1], f))

        self.svm.train_svm(trainvectors=self.X, labels=self.y, c='search', kernel='precomputed', gamma='search', jobs=10)


    def validate(self):
        ranking = {}
        y_real, y_pred = [], []
        for j, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep=10 * ' ', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens']
            q1_tree = query['tree'] if self.tree == 'tree' else query['subj_tree']
            q1_token2lemma = dict(zip(q1, query['lemmas']))
            q1_emb = self.encode(q1id, q1, self.fulldevidx, self.fulldevelmo)
            kq1 = self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, q1id, q1_tree, q1_emb, q1_token2lemma)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens']
                q2_tree = rel_question['tree'] if self.tree == 'tree' else rel_question['subj_tree']
                q2_token2lemma = dict(zip(q2, rel_question['lemmas']))
                q2_emb = self.encode(q2id, q2, self.fulldevidx, self.fulldevelmo)
                kq2 = self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, q2id, q2_tree, q2_emb, q2_token2lemma)

                X = []
                for j, c in enumerate(self.traindata):
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
                    X.append(k)

                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))
        parameter_settings = self.svm.return_parameter_settings(clf='svm')
        return ranking, y_real, y_pred, parameter_settings

    def test(self, testset, elmoidx, elmovec):
        self.testset = testset
        ranking = {}
        y_real, y_pred = [], []
        for j, q1id in enumerate(self.testset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testset), 2)
            print('Progress: ', percentage, j+1, sep=10 * ' ', end='\r')

            query = self.testset[q1id]
            q1 = query['tokens']
            q1_tree = query['tree'] if self.tree == 'tree' else query['subj_tree']
            q1_token2lemma = dict(zip(q1, query['lemmas']))
            q1_emb = self.encode(q1id, q1, elmoidx, elmovec)
            kq1 = self.memoize(q1id, q1_tree, q1_emb, q1_token2lemma, q1id, q1_tree, q1_emb, q1_token2lemma)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens']
                q2_tree = rel_question['tree'] if self.tree == 'tree' else rel_question['subj_tree']
                q2_token2lemma = dict(zip(q2, rel_question['lemmas']))
                q2_emb = self.encode(q2id, q2, elmoidx, elmovec)
                kq2 = self.memoize(q2id, q2_tree, q2_emb, q2_token2lemma, q2id, q2_tree, q2_emb, q2_token2lemma)

                X = []
                for j, c in enumerate(self.traindata):
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
                    X.append(k)

                score, pred_label = self.svm.score(X)
                y_pred.append(pred_label)

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
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