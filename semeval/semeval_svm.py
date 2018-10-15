__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': 'localhost', 'user': 'tcastrof'}
logger = logging.getLogger('tcpserver')

import copy
import _pickle as p
import features
import json
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import numpy as np
import os
import random
import re
import utils
import time

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

DATA_PATH='data'
FEATURE_PATH=os.path.join(DATA_PATH, 'trainfeatures.pickle')
FROBENIUS_PATH=os.path.join(DATA_PATH, 'trainfrobenius.pickle')
KERNEL_PATH=os.path.join(DATA_PATH, 'trainkernel.pickle')

LINEAR_MODEL_PATH=os.path.join(DATA_PATH, 'linear.model')
FROBENIUS_MODEL_PATH=os.path.join(DATA_PATH, 'frobenius.model')
TREE_MODEL_PATH=os.path.join(DATA_PATH, 'tree.model')

TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class SVM():
    def __init__(self, trainset, devset, testset):
        props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
        corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

        logging.info('Preparing development set...', extra=d)
        if not os.path.exists(DEV_PATH):
            self.devset = utils.prepare_corpus(devset, corenlp=corenlp, props=props)
            json.dump(self.devset, open(DEV_PATH, 'w'))
        else:
            self.devset = json.load(open(DEV_PATH))
        self.devdata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.devset)

        logging.info('Preparing trainset...', extra=d)
        if not os.path.exists(TRAIN_PATH):
            self.trainset = utils.prepare_corpus(trainset, corenlp=corenlp, props=props)
            json.dump(self.trainset, open(TRAIN_PATH, 'w'))
        else:
            self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info, extra=d)
        logging.info('Preparing test set...', extra=d)
        self.testset = utils.prepare_corpus(testset, corenlp=corenlp, props=props)

        corenlp.close()

        self.translation = features.init_translation(traindata=self.traindata,
                                                     vocabulary=self.vocabulary,
                                                     alpha=0.6,
                                                     sigma=0.6)

        self.trainidx, self.trainelmo, self.devidx, self.develmo = features.init_elmo()

        # self.embeddings, self.voc2id, self.id2voc = features.init_glove()


    def train_classifier(self, trainvectors, labels, c='1.0', kernel='linear', gamma='0.1', degree='1', class_weight='balanced', iterations=10, jobs=1):
            parameters = ['C', 'kernel', 'gamma', 'degree']
            if len(class_weight.split(':')) > 1: # dictionary
                class_weight = dict([label_weight.split(':') for label_weight in class_weight.split()])
            c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
            kernel_values = ['linear', 'poly'] if kernel == 'search' else [k for  k in kernel.split()]
            gamma_values = [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048] if gamma == 'search' else [float(x) for x in gamma.split()]
            degree_values = [1, 2] if degree == 'search' else [int(x) for x in degree.split()]
            grid_values = [c_values, kernel_values, gamma_values, degree_values]
            if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
                settings = {}
                for i, parameter in enumerate(parameters):
                    settings[parameter] = grid_values[i][0]
            else:
                param_grid = {}
                for i, parameter in enumerate(parameters):
                    param_grid[parameter] = grid_values[i]
                model = svm.SVC(probability=True)

                paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
                paramsearch.fit(trainvectors, labels)
                settings = paramsearch.best_params_
            # train an SVC classifier with the settings that led to the best performance
            model = svm.SVC(
                probability = True,
                C = settings[parameters[0]],
                kernel = settings[parameters[1]],
                gamma = settings[parameters[2]],
                degree = settings[parameters[3]],
                class_weight = class_weight,
                cache_size = 1000,
                verbose = 2
            )
            model.fit(trainvectors, labels)
            return model


    def select_traindata(self, data, n_samples):
        pos = list(filter(lambda x: x['label'] == 1, data))
        random.shuffle(pos)

        neg = list(filter(lambda x: x['label'] == 0, data))
        random.shuffle(neg)
        return pos[:int(n_samples/2.0)] + neg[:int(n_samples/2.0)]


    def train(self):
        raise NotImplementedError("Please Implement this method")

    def validate(self):
        raise NotImplementedError("Please Implement this method")


class BM25(SVM):
    def __init__(self, trainset, devset, testset):
        SVM.__init__(self, trainset, devset, testset)

    def train(self):
        logging.info('Setting BM25 model', extra=d)
        self.bm25_model, self.avg_idf, self.bm25_dct, self.bm25_qid_index = features.init_bm25(traindata=self.traindata, devdata=self.devdata, testdata=False)

    def validate(self):
        logging.info('Validating bm25.', extra=d)
        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(self.devset), 2)
            print('Progress: ', percentage, i+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens']
            scores = self.bm25_model.get_scores(self.bm25_dct.doc2bow(q1), self.avg_idf)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2score = scores[self.bm25_qid_index[q2id]]

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, q2score, q2id))

                logging.info('Finishing bm25 validation.')
        return ranking


class LinearSVM(SVM):
    def __init__(self, trainset, devset, testset):
        SVM.__init__(self, trainset, devset, testset)


    def __transform__(self, q1, q2):
        if type(q1) == list: q1 = ' '.join(q1)
        if type(q2) == list: q2 = ' '.join(q2)

        lcs = features.lcs(re.split('(\W)', q1), re.split('(\W)', q2))
        lcs1 = len(lcs[1].split())
        lcs2 = lcs[0]
        lcsub = features.lcsub(q1, q2)[0]
        jaccard = features.jaccard(q1, q2)
        containment_similarity = features.containment_similarities(q1, q2)
        # greedy_tiling = features.greedy_string_tiling(q1, q2)
        lmprob, trmprob, trlmprob, proctime = self.translation.score(q1.split(), q2.split())

        X = [lcs1, lcsub, jaccard, containment_similarity, lmprob]

        # ngram features
        for n in range(2, 5):
            ngram1 = ' '
            for gram in nltk.ngrams(q1.split(), n):
                ngram1 += 'x'.join(gram) + ' '

            ngram2 = ' '
            for gram in nltk.ngrams(q2.split(), n):
                ngram2 += 'x'.join(gram) + ' '

            lcs = features.lcs(re.split('(\W)', ngram1), re.split('(\W)', ngram2))
            X.append(len(lcs[1].split()))
            # X.append(lcs[0])
            X.append(features.lcsub(ngram1, ngram2)[0])
            X.append(features.jaccard(ngram1, ngram2))
            X.append(features.containment_similarities(ngram1, ngram2))

        return X


    def train(self):
        logging.info('Training svm.', extra=d)
        treekernel = features.TreeKernel(alpha=0, decay=1, ignore_leaves=True, smoothed=False)
        self.bm25_model, self.avg_idf, self.bm25_dct, self.bm25_qid_index = features.init_bm25(traindata=self.traindata, devdata=self.devdata, testdata=False)

        if not os.path.exists(FEATURE_PATH):
            X, y = [], []
            for i, query_question in enumerate(self.traindata):
                percentage = round(float(i+1) / len(self.traindata), 2)
                print('Preparing traindata: ', percentage, i+1, sep='\t', end='\r')
                q1, q2 = query_question['q1'], query_question['q2']
                x = self.__transform__(q1, q2)

                # bm25
                q1id = query_question['q1_id']
                q2id = query_question['q2_id']
                scores = self.bm25_model.get_scores(self.bm25_dct.doc2bow(q1), self.avg_idf)
                x.append(scores[self.bm25_qid_index[q2id]])

                # cosine
                # q1_lemma = query_question['subj_q1_lemmas']
                # q1_pos = query_question['subj_q1_pos']
                # q2_lemma = query_question['subj_q2_lemmas']
                # q2_pos = query_question['subj_q2_pos']
                # for n in range(1,5):
                #     x.append(features.cosine(' '.join(q1), ' '.join(q2), n=n))
                #     x.append(features.cosine(' '.join(q1_lemma), ' '.join(q2_lemma), n=n))
                #     x.append(features.cosine(' '.join(q1_pos), ' '.join(q2_pos), n=n))

                # tree kernels
                q1_tree, q2_tree = utils.parse_tree(query_question['q1_tree']), utils.parse_tree(query_question['q2_tree'])
                q1_tree, q2_tree = treekernel.similar_terminals(q1_tree, q2_tree)
                x.append(treekernel(q1_tree, q2_tree))

                # frobenius norm
                q1_emb = self.trainelmo.get(str(self.trainidx[query_question['q1_id']]))
                q2_emb = self.trainelmo.get(str(self.trainidx[query_question['q2_id']]))
                x.append(features.frobenius_norm(q1_emb, q2_emb))

                X.append(x)
                y.append(query_question['label'])

            p.dump(list(zip(X, y)), open(FEATURE_PATH, 'wb'))
        else:
            f = p.load(open(FEATURE_PATH, 'rb'))
            X = list(map(lambda x: x[0], f))
            y = list(map(lambda x: x[1], f))

        # scale features
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        X_array = np.array(X)
        self.scaler.fit(X_array)
        X = self.scaler.transform(X_array).tolist()

        if not os.path.exists(LINEAR_MODEL_PATH):
            self.model = self.train_classifier(
                trainvectors=X,
                labels=y,
                c='search',
                kernel='search',
                gamma='search',
                degree='search',
                jobs=10
            )
            p.dump(self.model, open(LINEAR_MODEL_PATH, 'wb'))
        else:
            self.model = p.load(open(LINEAR_MODEL_PATH, 'rb'))
        logging.info('Finishing to train svm.')


    def validate(self):
        logging.info('Validating svm.', extra=d)
        treekernel = features.TreeKernel(alpha=0, decay=1, ignore_leaves=True, smoothed=False)
        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(self.devset), 2)
            print('Progress: ', percentage, i+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens']
            scores = self.bm25_model.get_scores(self.bm25_dct.doc2bow(q1), self.avg_idf)
            q1_lemma = query['subj_lemmas_full']
            q1_pos = query['subj_pos_full']
            q1_tree = utils.parse_tree(query['subj_tree'])
            q1_emb = self.develmo.get(str(self.devidx[q1id]))

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens']
                X = self.__transform__(q1, q2)
                # bm25
                X.append(scores[self.bm25_qid_index[q2id]])

                # cosine
                # q2_lemma = rel_question['subj_lemma_full']
                # q2_pos = rel_question['subj_pos_full']
                # for n in range(1,5):
                #     X.append(features.cosine(' '.join(q1), ' '.join(q2), n=n))
                #     X.append(features.cosine(' '.join(q1_lemma), ' '.join(q2_lemma), n=n))
                #     X.append(features.cosine(' '.join(q1_pos), ' '.join(q2_pos), n=n))

                # tree kernel
                q2_tree = utils.parse_tree(rel_question['subj_tree'])
                # q1_tree, q2_tree = treekernel.similar_terminals(q1_tree, q2_tree)
                X.append(treekernel(q1_tree, q2_tree))

                # frobenius norm
                q2_emb = self.develmo.get(str(self.devidx[q2id]))
                X.append(features.frobenius_norm(q1_emb, q2_emb))

                # scale
                X = self.scaler.transform(X.toarray()).tolist()

                score = self.model.decision_function([X])[0]
                pred_label = self.model.predict([X])[0]
                y_pred.append(pred_label)

                # if pred_label == 1:
                #     score += 100
                # else:
                #     score -= 100

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        with open('data/ranking.txt', 'w') as f:
            for q1id in ranking:
                for row in ranking[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))

        logging.info('Finishing to validate svm.', extra=d)
        return ranking, y_real, y_pred


class TreeSVM(SVM):
    def __init__(self, trainset, devset, testset):
        SVM.__init__(self, trainset, devset, testset)

        # self.traindata = self.select_traindata(self.traindata, 1500)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info, extra=d)
        self.memoization = {}


    def memoize(self, q1id, q1, q1_emb, q2id, q2, q2_emb, kernel):
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

        k = kernel(q1, q2, q1_emb, q2_emb)
        self.memoization[q1id][q2id] = k
        self.memoization[q2id][q1id] = k

        return k


    def train(self):
        logging.info('Training tree svm.', extra=d)
        treekernel = features.TreeKernel()

        if not os.path.exists(KERNEL_PATH):
            X, y = [], []
            for i, q in enumerate(self.traindata):
                percentage = round(float(i+1) / len(self.traindata), 2)
                x = []
                q1id, q2id = q['q1_id'], q['q2_id']
                # trees
                q1, q2 = utils.parse_tree(q['subj_q1_tree']), utils.parse_tree(q['subj_q2_tree'])
                # elmo vectors
                q1_emb = self.trainelmo.get(str(self.trainidx[q1id]))
                q2_emb = self.trainelmo.get(str(self.trainidx[q2id]))

                # similar terminals
                q1, q2 = treekernel.similar_terminals(q1, q2)
                for j, c in enumerate(self.traindata):
                    c1id, c2id = c['q1_id'], c['q2_id']
                    # trees
                    c1, c2 = utils.parse_tree(c['subj_q1_tree']), utils.parse_tree(c['subj_q2_tree'])
                    # elmo vectors
                    c1_emb = self.trainelmo.get(str(self.trainidx[c1id]))
                    c2_emb = self.trainelmo.get(str(self.trainidx[c2id]))
                    # similar terminals
                    c1, c2 = treekernel.similar_terminals(c1, c2)
                    k11 = self.memoize(q1id, q1, q1_emb, q1id, q1, q1_emb, treekernel)
                    k12 = self.memoize(c1id, c1, c1_emb, c1id, c1, c1_emb, treekernel)
                    k1 = float(self.memoize(q1id, q1, q1_emb, c1id, c1, c1_emb, treekernel)) / np.sqrt(k11 * k12) # normalized

                    k21 = self.memoize(q2id, q2, q2_emb, q2id, q2, q2_emb, treekernel)
                    k22 = self.memoize(c2id, c2, c2_emb, c2id, c2, c2_emb, treekernel)
                    k2 = float(self.memoize(q2id, q2, q2_emb, c2id, c2, c2_emb, treekernel)) / np.sqrt(k21 * k22) # normalized

                    k = k1 + k2
                    x.append(k)
                print('Preparing kernel: ', percentage, i+1, sep='\t', end='\r')
                X.append(x)
                y.append(q['label'])
            p.dump(list(zip(X, y)), open(KERNEL_PATH, 'wb'))
            X = np.array(X)
        else:
            f = p.load(open(KERNEL_PATH, 'rb'))
            X = np.array([x[0] for x in f])
            y = list(map(lambda x: x[1], f))

        if not os.path.exists(TREE_MODEL_PATH):
            self.model = self.train_classifier(
                trainvectors=X,
                labels=y,
                c='search',
                kernel='precomputed',
                gamma='search',
                # degree='search',
                jobs=10
            )
            p.dump(self.model, open(TREE_MODEL_PATH, 'wb'))
        else:
            self.model = p.load(open(TREE_MODEL_PATH, 'rb'))
        logging.info('Finishing to train tree svm.', extra=d)


    def validate(self):
        logging.info('Validating tree svm.', extra=d)
        treekernel = features.TreeKernel()
        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(self.devset), 2)

            query = self.devset[q1id]
            q1_tree = utils.parse_tree(query['subj_str_tree'])

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                # tree kernel
                q2_tree = utils.parse_tree(rel_question['subj_str_tree'])
                # similar terminals
                q1_tree, q2_tree = treekernel.similar_terminals(q1_tree, q2_tree)
                # elmo vectors
                q1_emb = self.develmo.get(str(self.devidx[q1id]))
                q2_emb = self.develmo.get(str(self.devidx[q2id]))

                X = []
                for j, trainrow in enumerate(self.traindata):
                    c1id, c2id = trainrow['q1_id'], trainrow['q2_id']
                    c1_tree, c2_tree = utils.parse_tree(trainrow['subj_q1_tree']), utils.parse_tree(trainrow['subj_q2_tree'])
                    # similar terminals
                    c1_tree, c2_tree = treekernel.similar_terminals(c1_tree, c2_tree)
                    # elmo vectors
                    c1_emb = self.trainelmo.get(str(self.trainidx[c1id]))
                    c2_emb = self.trainelmo.get(str(self.trainidx[c2id]))

                    k11 = self.memoize(q1id, q1_tree, q1_emb, q1id, q1_tree, q1_emb, treekernel)
                    k12 = self.memoize(c1id, c1_tree, c1_emb, c1id, c1_tree, c1_emb, treekernel)
                    k1 = float(self.memoize(q1id, q1_tree, q1_emb, c1id, c1_tree, c1_emb, treekernel)) / np.sqrt(k11 * k12) # normalized

                    k21 = self.memoize(q2id, q2_tree, q2_emb, q2id, q2_tree, q2_emb, treekernel)
                    k22 = self.memoize(c2id, c2_tree, c2_emb, c2id, c2_tree, c2_emb, treekernel)
                    k2 = float(self.memoize(q2id, q2_tree, q2_emb, c2id, c2_tree, c2_emb, treekernel)) / np.sqrt(k21 * k22) # normalized

                    kernel = k1 + k2
                    X.append(kernel)
                print('Progress: ', percentage, i+1, sep='\t', end='\r')

                score = self.model.decision_function([X])[0]
                pred_label = self.model.predict([X])[0]
                y_pred.append(pred_label)

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        with open('data/treeranking.txt', 'w') as f:
            for qid in ranking:
                for row in ranking[qid]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(qid), str(row[2]), str(0), str(row[1]), label, '\n']))

        logging.info('Finishing to validate tree svm.', extra=d)
        return ranking, y_real, y_pred


class ELMoSVM(SVM):
    def __init__(self, trainset, devset, testset):
        SVM.__init__(self, trainset, devset, testset)
        self.traindata = self.select_traindata(self.traindata, 1000)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info, extra=d)
        self.memoization = {}


    def memoize(self, q1id, q1, q2id, q2):
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

        frob = features.frobenius_norm(query_emb=q1, question_emb=q2)
        self.memoization[q1id][q2id] = frob
        self.memoization[q2id][q1id] = frob

        return frob


    def train(self):
        logging.info('Training frobenius svm.', extra=d)

        if not os.path.exists(FROBENIUS_PATH):
            X, y = [], []
            for i, q in enumerate(self.traindata):
                proctime = []
                percentage = round(float(i+1) / len(self.traindata), 2)
                x = []
                # frobenius norm
                q1id, q2id = q['q1_id'], q['q2_id']
                q1_emb = self.trainelmo.get(str(self.trainidx[q['q1_id']]))
                q2_emb = self.trainelmo.get(str(self.trainidx[q['q2_id']]))
                for j, c in enumerate(self.traindata):
                    c1id, c2id = c['q1_id'], c['q2_id']
                    c1_emb = self.trainelmo.get(str(self.trainidx[c['q1_id']]))
                    c2_emb = self.trainelmo.get(str(self.trainidx[c['q2_id']]))

                    start = time.time()
                    frob11 = self.memoize(q1id, q1_emb, q1id, q1_emb)
                    frob12 = self.memoize(c1id, c1_emb, c1id, c1_emb)
                    frob1 = float(self.memoize(q1id, q1_emb, c1id, c1_emb)) / np.sqrt(frob11 * frob12) # normalized

                    frob21 = self.memoize(q2id, q2_emb, q2id, q2_emb)
                    frob22 = self.memoize(c2id, c2_emb, c2id, c2_emb)
                    frob2 = float(self.memoize(q2id, q2_emb, c2id, c2_emb)) / np.sqrt(frob21 * frob22) # normalized
                    x.append(frob1+frob2)
                    end = time.time()
                    proctime.append(round(end-start, 4))
                print('Preparing traindata: ', percentage, i+1, round(np.mean(proctime), 4), sep='\t', end='\r')

                X.append(x)
                y.append(q['label'])

            p.dump(list(zip(X, y)), open(FROBENIUS_PATH, 'wb'))
            X = np.array(X)
        else:
            f = p.load(open(FROBENIUS_PATH, 'rb'))
            X = np.array([x[0] for x in f])
            y = list(map(lambda x: x[1], f))

        if not os.path.exists(FROBENIUS_MODEL_PATH):
            self.model = self.train_classifier(
                trainvectors=X,
                labels=y,
                c='search',
                kernel='precomputed',
                gamma='search',
                jobs=10
            )
            p.dump(self.model, open(FROBENIUS_MODEL_PATH, 'wb'))
        else:
            self.model = p.load(open(FROBENIUS_MODEL_PATH, 'rb'))
        logging.info('Finishing to train frobenius svm.', extra=d)


    def validate(self):
        logging.info('Validating frobenius svm.', extra=d)
        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(self.devset), 2)

            query = self.devset[q1id]
            q1_emb = self.develmo.get(str(self.devidx[q1id]))

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                X = []
                q2_emb = self.develmo.get(str(self.devidx[q2id]))
                for j, c in enumerate(self.traindata):
                    c1id, c2id = c['q1_id'], c['q2_id']
                    c1_emb = self.trainelmo.get(str(self.trainidx[c['q1_id']]))
                    c2_emb = self.trainelmo.get(str(self.trainidx[c['q2_id']]))

                    frob11 = self.memoize(q1id, q1_emb, q1id, q1_emb)
                    frob12 = self.memoize(c1id, c1_emb, c1id, c1_emb)
                    frob1 = float(self.memoize(q1id, q1_emb, c1id, c1_emb)) / np.sqrt(frob11 * frob12) # normalized

                    frob21 = self.memoize(q2id, q2_emb, q2id, q2_emb)
                    frob22 = self.memoize(c2id, c2_emb, c2id, c2_emb)
                    frob2 = float(self.memoize(q2id, q2_emb, c2id, c2_emb)) / np.sqrt(frob21 * frob22) # normalized
                    X.append(frob1+frob2)
                print('Progress: ', percentage, i+1, sep='\t', end='\r')

                score = self.model.decision_function([X])[0]
                pred_label = self.model.predict([X])[0]
                y_pred.append(pred_label)

                # if pred_label == 1:
                #     score += 100
                # else:
                #     score -= 100

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

        with open('data/frobranking.txt', 'w') as f:
            for qid in ranking:
                for row in ranking[qid]:
                    f.write('\t'.join([str(qid), str(row[2]), str(row[1]), str(row[0]), '\n']))

        logging.info('Finishing to validate frobenius svm.', extra=d)
        return ranking, y_real, y_pred


if __name__ == '__main__':
    logging.info('Load corpus', extra=d)
    trainset, devset = load.run()

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # Linear SVM
    linear = LinearSVM(trainset, devset, [])
    linear.train()
    # Tree SVM
    semeval = TreeSVM(trainset, devset, [])
    semeval.train()
    p.dump(semeval.memoization, open('data/treememoization.pickle', 'wb'))
    # BM25
    bm25 = BM25(trainset, devset, [])
    bm25.train()

    # Linear performance
    linear_ranking, y_real, y_pred = linear.validate()
    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, copy.copy(linear_ranking))
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation Linear')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)
    print(10 * '-')

    # Tree performance
    tree_ranking, y_real, y_pred = semeval.validate()
    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, copy.copy(tree_ranking))
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation Tree')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)
    print(10 * '-')

    # BM25 performance
    bm25_ranking = bm25.validate()
    map_baseline, map_model = utils.evaluate(devgold, copy.copy(bm25_ranking))

    print('Evaluation BM25')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    ranking = {}
    for qid in linear_ranking:
        lrank = linear_ranking[qid]
        trank = tree_ranking[qid]

        ranking[qid] = []
        for row1 in lrank:
            for row2 in trank:
                if row1[2] == row2[2]:
                    score = (0.6 * row1[1]) + (0.4 * row2[1])
                    ranking[qid].append((row1[0], score, row1[2]))


    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, copy.copy(ranking))

    print('Evaluation Linear+Tree Weighted')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')


    ranking = {}
    for qid in linear_ranking:
        lrank = linear_ranking[qid]
        trank = tree_ranking[qid]

        ranking[qid] = []
        for row1 in lrank:
            for row2 in trank:
                if row1[2] == row2[2]:
                    score = row1[1] + row2[1]
                    ranking[qid].append((row1[0], score, row1[2]))


    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, copy.copy(ranking))

    print('Evaluation Linear+Tree')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')
