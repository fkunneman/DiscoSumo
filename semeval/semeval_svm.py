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
import preprocessing
import random
import re
import utils

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression

from semeval_cos import SemevalQuestionCosine

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

DATA_PATH='data'
FEATURE_PATH=os.path.join(DATA_PATH, 'trainfeatures.pickle')
KERNEL_PATH=os.path.join(DATA_PATH, 'trainkernel.pickle')

TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class SemevalModel():
    def __init__(self):
        if not os.path.exists(DEV_PATH):
            preprocessing.run()

        logging.info('Preparing development set...', extra=d)
        self.devset = json.load(open(DEV_PATH))
        self.devdata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.devset)

        logging.info('Preparing trainset...', extra=d)
        self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info, extra=d)

        self.translation = features.init_translation(traindata=self.trainset, alpha=0.7, sigma=0.3)

        logging.info('Preparing SimBOW...', extra=d)
        self.simbow = SemevalQuestionCosine()
        self.simbow.train()

        self.trainidx, self.trainelmo, self.devidx, self.develmo = features.init_elmo()
        self.fulltrainidx, self.fulltrainelmo, self.fulldevidx, self.fulldevelmo = features.init_elmo(stop=False)
        self.word2vec = features.init_word2vec()
        # self.embeddings, self.voc2id, self.id2voc = features.init_glove()


    def train_svm(self, trainvectors, labels, c='1.0', kernel='linear', gamma='0.1', degree='1', class_weight='balanced', iterations=10, jobs=1):
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

    def train_regression(self, trainvectors, labels, c='1.0', penalty='l1', tol='1e-4', solver='saga', iterations=10, jobs=1):
        parameters = ['C', 'penalty', 'tol']
        c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        penalty_values = ['l1', 'l2'] if penalty == 'search' else [k for  k in penalty.split()]
        tol_values = [1, 0.1, 0.01, 0.001, 0.0001] if tol == 'search' else [float(x) for x in tol.split()]
        grid_values = [c_values, penalty_values, tol_values]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = LogisticRegression(solver=solver)

            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_
        # train an SVC classifier with the settings that led to the best performance
        model = LogisticRegression(
            C = settings[parameters[0]],
            penalty = settings[parameters[1]],
            tol = settings[parameters[2]],
            solver= solver,
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


class BM25(SemevalModel):
    def __init__(self):
        SemevalModel.__init__(self)

    def train(self):
        logging.info('Setting BM25 model', extra=d)
        self.bm25_model, self.avg_idf, self.bm25_qid_index = features.init_bm25(traindata=self.trainset, devdata=self.devset, testdata=[])

    def validate(self):
        logging.info('Validating bm25.', extra=d)
        ranking = {}
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(self.devset), 2)
            print('Progress: ', percentage, i+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2score = self.bm25_model.get_score(q1, self.bm25_qid_index[q2id], self.avg_idf)

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, q2score, q2id))

                logging.info('Finishing bm25 validation.')
        return ranking


class LDA(SemevalModel):
    def __init__(self):
        SemevalModel.__init__(self)

    def train(self):
        logging.info('Setting LDA model', extra=d)
        self.lda_model, self.lda_vectorizer = features.init_lda(traindata=self.traindata,n_topics=50)

    def validate(self):
        logging.info('Validating LDA.', extra=d)
        ranking = {}
        for i, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(self.devset), 2)
            print('Progress: ', percentage, i+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = ' '.join(query['tokens'])
            q1_vector = self.lda_vectorizer.transform([q1]).toarray()
            q1_topicvector = []
            nonzero = np.count_nonzero(q1_vector)
            for mc in self.lda_model.components_:
                q1_topicvector.append(np.sum(q1_vector * mc) / nonzero)
            q1_topicvector_array = np.array(q1_topicvector)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = ' '.join(rel_question['tokens'])
                rel_question_vector = self.lda_vectorizer.transform([q2]).toarray()
                q2_topicvector = []
                nonzero = np.count_nonzero(q2_vector)
                for mc in self.lda_model.components_:
                    q2_topicvector.append(np.sum(q2_vector * mc) / nonzero)
                q2score = np.sum(q1_topicvector_array * np.array(q2_topicvector))

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                ranking[q1id].append((real_label, q2score, q2id))

                logging.info('Finishing lda validation.')
        return ranking


class LinearSVM(SemevalModel):
    def __init__(self):
        SemevalModel.__init__(self)


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

        X = [lcs1, lcsub, jaccard, containment_similarity]

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


    def get_features(self, q1id, q1, q2id, q2, set='train'):
        X = []
        if set == 'train':
            q1_elmo = self.trainelmo.get(str(self.trainidx[q1id]))
            q2_elmo = self.trainelmo.get(str(self.trainidx[q2id]))
        else:
            q1_elmo = self.develmo.get(str(self.devidx[q1id]))
            q2_elmo = self.develmo.get(str(self.devidx[q2id]))

        q1_w2v = features.encode(q1, self.word2vec)
        q1_elmo_bottom = [np.concatenate([q1_w2v[i], q1_elmo[0][i]]) for i in range(len(q1_w2v))]
        q1_elmo_middle = [np.concatenate([q1_w2v[i], q1_elmo[1][i]]) for i in range(len(q1_w2v))]
        q1_elmo_top = [np.concatenate([q1_w2v[i], q1_elmo[2][i]]) for i in range(len(q1_w2v))]

        q2_w2v = features.encode(q2, self.word2vec)
        q2_elmo_bottom = [np.concatenate([q2_w2v[i], q2_elmo[0][i]]) for i in range(len(q2_w2v))]
        q2_elmo_middle = [np.concatenate([q2_w2v[i], q2_elmo[1][i]]) for i in range(len(q2_w2v))]
        q2_elmo_top = [np.concatenate([q2_w2v[i], q2_elmo[2][i]]) for i in range(len(q2_w2v))]


        # X.append(self.simbow.score(q1, q1_w2v, q2, q2_w2v))
        X.append(self.simbow.score(q1, q1_elmo_bottom, q2, q2_elmo_bottom))
        X.append(self.simbow.score(q1, q1_elmo_middle, q2, q2_elmo_middle))
        X.append(self.simbow.score(q1, q1_elmo_top, q2, q2_elmo_top))
        return X


    def train(self):
        logging.info('Training svm.', extra=d)
        treekernel = features.TreeKernel(alpha=0, decay=1, ignore_leaves=True, smoothed=False)
        self.bm25_model, self.avg_idf, self.bm25_qid_index = features.init_bm25(traindata=self.trainset, devdata=self.devset, testdata=[])

        if not os.path.exists(FEATURE_PATH):
            X, y = [], []
            for i, query_question in enumerate(self.traindata):
                percentage = round(float(i+1) / len(self.traindata), 2)
                print('Preparing traindata: ', percentage, i+1, sep='\t', end='\r')
                q1id = query_question['q1_id']
                q2id = query_question['q2_id']
                q1, q2 = query_question['q1'], query_question['q2']
                # x = self.get_features(q1id, q1, q2id, q2)
                x = []
                # x = self.__transform__(q1, q2)
                #
                # # elmo and word2vec embeddings
                q1_elmo = self.trainelmo.get(str(self.trainidx[q1id]))
                q1_w2v = features.encode(q1, self.word2vec)
                q1_emb = [np.concatenate([q1_w2v[i], q1_elmo[i]]) for i in range(len(q1_w2v))]

                q2_elmo = self.trainelmo.get(str(self.trainidx[q2id]))
                q2_w2v = features.encode(q2, self.word2vec)
                q2_emb = [np.concatenate([q2_w2v[i], q2_elmo[i]]) for i in range(len(q2_w2v))]

                # # translation
                # lmprob, trmprob, trlmprob, proctime = self.translation.score_embeddings(q1, q1_emb, q2, q2_emb)
                # x.append(trlmprob)
                #
                # # bm25
                # bm25_score = self.bm25_model.get_score(q1, self.bm25_qid_index[q2id], self.avg_idf)
                # x.append(bm25_score)
                #
                # # cosine
                # q1_lemma = query_question['q1_lemmas']
                # q1_pos = query_question['q1_pos']
                # q2_lemma = query_question['q2_lemmas']
                # q2_pos = query_question['q2_pos']
                # for n in range(1,5):
                #     try:
                #         x.append(features.cosine(' '.join(q1), ' '.join(q2), n=n))
                #     except:
                #         x.append(0.0)
                #     try:
                #         x.append(features.cosine(' '.join(q1_lemma), ' '.join(q2_lemma), n=n))
                #     except:
                #         x.append(0.0)
                #     try:
                #         x.append(features.cosine(' '.join(q1_pos), ' '.join(q2_pos), n=n))
                #     except:
                #         x.append(0.0)
                #
                # # tree kernels
                # q1_token2lemma = dict(zip(query_question['q1_full'], query_question['q1_lemmas']))
                # q2_token2lemma = dict(zip(query_question['q2_full'], query_question['q2_lemmas']))
                # q1_tree, q2_tree = utils.parse_tree(query_question['q1_tree'], q1_token2lemma), utils.parse_tree(query_question['q2_tree'], q2_token2lemma)
                # q1_tree, q2_tree = treekernel.similar_terminals(q1_tree, q2_tree)
                # x.append(treekernel(q1_tree, q2_tree))
                #
                # # frobenius norm
                # x.append(features.frobenius_norm(q1_emb, q2_emb))
                #
                # # softcosine
                simbow = self.simbow.score(q1, q1_emb, q2, q2_emb)
                x.append(simbow)

                for comment in query_question['comments']:
                    q3id = comment['id']
                    q3 = comment['tokens']
                    simbow_q1q3, simbow_q2q3 = 0, 0
                    if len(q3) > 0:
                        # x.extend(self.get_features(q1id, q1, q3id, q3))
                        q3_elmo = self.trainelmo.get(str(self.trainidx[q3id]))
                        q3_w2v = features.encode(q3, self.word2vec)
                        q3_emb = [np.concatenate([q3_w2v[i], q3_elmo[i]]) for i in range(len(q3_w2v))]
                        simbow_q1q3 = self.simbow.score(q1, q1_emb, q3, q3_emb)
                        # simbow_q2q3 = self.simbow.score(q2, q2_emb, q3, q3_emb)
                        # lmprob, trmprob, trlmprob, proctime = self.translation.score_embeddings(q1, q1_emb, q3, q3_emb)
                        # bm25_score = self.bm25_model.get_score(q1, self.bm25_qid_index[comment['id']], self.avg_idf)

                    # x.append(trlmprob)
                    # x.append(bm25_score)
                    x.append(simbow_q1q3)
                    # x.append(simbow_q2q3)

                X.append(x)
                y.append(query_question['label'])

            p.dump(list(zip(X, y)), open(FEATURE_PATH, 'wb'))
        else:
            f = p.load(open(FEATURE_PATH, 'rb'))
            X = list(map(lambda x: x[0], f))
            y = list(map(lambda x: x[1], f))

        # scale features
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        clf = LassoCV(cv=10)
        self.feat_selector = SelectFromModel(clf)
        self.feat_selector.fit(X, y)
        X = self.feat_selector.transform(X)

        self.model = self.train_svm(
            trainvectors=X,
            labels=y,
            c='search',
            kernel='search',
            gamma='search',
            degree='search',
            jobs=4
        )
        # self.model = self.train_regression(trainvectors=X, labels=y, c='search', penalty='search', tol='search')
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
            q1 = query['tokens_proc']
            # q1_lemma = query['lemmas']
            # q1_pos = query['pos']
            # q1_token2lemma = dict(zip(query['tokens'], query['lemmas']))
            # q1_tree = utils.parse_tree(query['subj_tree'], q1_token2lemma)

            q1_elmo = self.develmo.get(str(self.devidx[q1id]))
            q1_w2v = features.encode(q1, self.word2vec)
            q1_emb = [np.concatenate([q1_w2v[i], q1_elmo[i]]) for i in range(len(q1_w2v))]

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens_proc']
                # X = self.get_features(q1id, q1, q2id, q2, set='dev')
                # X = self.__transform__(q1, q2)
                X = []

                q2_elmo = self.develmo.get(str(self.devidx[q2id]))
                q2_w2v = features.encode(q2, self.word2vec)
                q2_emb = [np.concatenate([q2_w2v[i], q2_elmo[i]]) for i in range(len(q2_w2v))]

                # # translation
                # lmprob, trmprob, trlmprob, proctime = self.translation.score_embeddings(q1, q1_emb, q2, q2_emb)
                # X.append(trlmprob)
                #
                # # bm25
                # bm25_score = self.bm25_model.get_score(q1, self.bm25_qid_index[q2id], self.avg_idf)
                # X.append(bm25_score)
                #
                # # cosine
                # q2_lemma = rel_question['lemmas']
                # q2_pos = rel_question['pos']
                # for n in range(1,5):
                #     try:
                #         X.append(features.cosine(' '.join(q1), ' '.join(q2), n=n))
                #     except:
                #         X.append(0.0)
                #     try:
                #         X.append(features.cosine(' '.join(q1_lemma), ' '.join(q2_lemma), n=n))
                #     except:
                #         X.append(0.0)
                #     try:
                #         X.append(features.cosine(' '.join(q1_pos), ' '.join(q2_pos), n=n))
                #     except:
                #         X.append(0.0)
                #
                # # tree kernel
                # q2_token2lemma = dict(zip(rel_question['tokens'], rel_question['lemmas']))
                # q2_tree = utils.parse_tree(rel_question['subj_tree'], q2_token2lemma)
                # q1_tree, q2_tree = treekernel.similar_terminals(q1_tree, q2_tree)
                # X.append(treekernel(q1_tree, q2_tree))
                #
                # # frobenius norm
                # X.append(features.frobenius_norm(q1_emb, q2_emb))

                # softcosine
                simbow = self.simbow.score(q1, q1_emb, q2, q2_emb)
                X.append(simbow)

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = comment['tokens_proc']
                    simbow_q1q3, simbow_q2q3 = 0, 0
                    if len(q3) > 0:
                        # X.extend(self.get_features(q1id, q1, q3id, q3, set='dev'))
                        q3_elmo = self.develmo.get(str(self.devidx[comment['id']]))
                        q3_w2v = features.encode(q3, self.word2vec)
                        q3_emb = [np.concatenate([q3_w2v[i], q3_elmo[i]]) for i in range(len(q3_w2v))]
                        simbow_q1q3 = self.simbow.score(q1, q1_emb, q3, q3_emb)
                        # simbow_q2q3 = self.simbow.score(q2, q2_emb, q3, q3_emb)
                        # bm25_score = self.bm25_model.get_score(q1, self.bm25_qid_index[comment['id']], self.avg_idf)
                        # lmprob, trmprob, trlmprob, proctime = self.translation.score_embeddings(q1, q1_emb, q3, q3_emb)
                    # X.append(trlmprob)
                    # X.append(bm25_score)
                    X.append(simbow_q1q3)
                    # X.append(simbow_q2q3)

                # scale
                X = self.scaler.transform([X])
                # feature selection
                X = self.feat_selector.transform(X)

                score = self.model.decision_function(X)[0]
                pred_label = self.model.predict(X)[0]
                y_pred.append(pred_label)

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


class TreeSVM(SemevalModel):
    def __init__(self):
        SemevalModel.__init__(self)

        self.traindata = self.select_traindata(self.traindata, 1500)
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
                q1_token2lemma = dict(zip(q['q1_full'], q['q1_lemmas']))
                q2_token2lemma = dict(zip(q['q2_full'], q['q2_lemmas']))
                q1 = utils.binarize(utils.parse_tree(q['q1_tree'], q1_token2lemma))
                q2 = utils.binarize(utils.parse_tree(q['q2_tree'], q2_token2lemma))

                # word2vec and elmo vectors
                q1_w2v = features.encode(q['q1_full'], self.word2vec)
                q1_elmo = self.fulltrainelmo.get(str(self.fulltrainidx[q1id]))
                q1_emb = [np.concatenate([q1_w2v[i], q1_elmo[i]]) for i in range(len(q1_w2v))]

                q2_w2v = features.encode(q['q2_full'], self.word2vec)
                q2_elmo = self.fulltrainelmo.get(str(self.fulltrainidx[q2id]))
                q2_emb = [np.concatenate([q2_w2v[i], q2_elmo[i]]) for i in range(len(q2_w2v))]

                q1, q2 = treekernel.similar_terminals(q1, q2)
                for j, c in enumerate(self.traindata):
                    c1id, c2id = c['q1_id'], c['q2_id']
                    # trees
                    c1_token2lemma = dict(zip(c['q1_full'], c['q1_lemmas']))
                    c2_token2lemma = dict(zip(c['q2_full'], c['q2_lemmas']))
                    c1 = utils.binarize(utils.parse_tree(c['q1_tree'], c1_token2lemma))
                    c2 = utils.binarize(utils.parse_tree(c['q2_tree'], c2_token2lemma))
                    # word2vec vectors
                    c1_w2v = features.encode(c['q1_full'], self.word2vec)
                    c1_elmo = self.fulltrainelmo.get(str(self.fulltrainidx[c1id]))
                    c1_emb = [np.concatenate([c1_w2v[i], c1_elmo[i]]) for i in range(len(c1_w2v))]

                    c2_w2v = features.encode(c['q2_full'], self.word2vec)
                    c2_elmo = self.fulltrainelmo.get(str(self.fulltrainidx[c2id]))
                    c2_emb = [np.concatenate([c2_w2v[i], c2_elmo[i]]) for i in range(len(c2_w2v))]

                    c1, c2 = treekernel.similar_terminals(c1, c2)
                    kq1 = self.memoize(q1id, q1, q1_emb, q1id, q1, q1_emb, treekernel)
                    kc1 = self.memoize(c1id, c1, c1_emb, c1id, c1, c1_emb, treekernel)
                    kq1c1 = float(self.memoize(q1id, q1, q1_emb, c1id, c1, c1_emb, treekernel)) / np.sqrt(kq1 * kc1) # normalized

                    kq2 = self.memoize(q2id, q2, q2_emb, q2id, q2, q2_emb, treekernel)
                    kc2 = self.memoize(c2id, c2, c2_emb, c2id, c2, c2_emb, treekernel)
                    kq2c2 = float(self.memoize(q2id, q2, q2_emb, c2id, c2, c2_emb, treekernel)) / np.sqrt(kq2 * kc2) # normalized

                    # kq1c2 = float(self.memoize(q1id, q1, q1_emb, c2id, c2, c2_emb, treekernel)) / np.sqrt(kq1 * kc2) # normalized
                    # kq2c1 = float(self.memoize(q2id, q2, q2_emb, c1id, c1, c1_emb, treekernel)) / np.sqrt(kq2 * kc1) # normalized

                    k = kq1c1 + kq2c2
                    x.append(k)
                    print('Preparing kernel: ', percentage, i+1, j+1, sep='\t', end='\r')
                X.append(x)
                y.append(q['label'])
            p.dump(list(zip(X, y)), open(KERNEL_PATH, 'wb'))
            X = np.array(X)
        else:
            f = p.load(open(KERNEL_PATH, 'rb'))
            X = np.array([x[0] for x in f])
            y = list(map(lambda x: x[1], f))

        self.model = self.train_svm(
            trainvectors=X,
            labels=y,
            c='search',
            kernel='precomputed',
            gamma='search',
            jobs=4
        )
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
            q1_token2lemma = dict(zip(query['tokens'], query['lemmas']))
            q1_tree = utils.binarize(utils.parse_tree(query['tree'], q1_token2lemma))

            q1_w2v = features.encode(query['tokens'], self.word2vec)
            q1_elmo = self.fulldevelmo.get(str(self.fulldevidx[q1id]))
            q1_emb = [np.concatenate([q1_w2v[i], q1_elmo[i]]) for i in range(len(q1_w2v))]

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                # tree kernel
                q2_token2lemma = dict(zip(rel_question['tokens'], rel_question['lemmas']))
                q2_tree = utils.binarize(utils.parse_tree(rel_question['tree'], q2_token2lemma))

                # word2vec vectors
                q2_w2v = features.encode(rel_question['tokens'], self.word2vec)
                q2_elmo = self.fulldevelmo.get(str(self.fulldevidx[q2id]))
                q2_emb = [np.concatenate([q2_w2v[i], q2_elmo[i]]) for i in range(len(q2_w2v))]

                q1_tree, q2_tree = treekernel.similar_terminals(q1_tree, q2_tree)

                X = []
                for j, trainrow in enumerate(self.traindata):
                    c1id, c2id = trainrow['q1_id'], trainrow['q2_id']
                    c1_token2lemma = dict(zip(trainrow['q1_full'], trainrow['q1_lemmas']))
                    c2_token2lemma = dict(zip(trainrow['q2_full'], trainrow['q2_lemmas']))
                    c1_tree = utils.binarize(utils.parse_tree(trainrow['q1_tree'], c1_token2lemma))
                    c2_tree = utils.binarize(utils.parse_tree(trainrow['q2_tree'], c2_token2lemma))

                    # word2vec vectors
                    c1_w2v = features.encode(trainrow['q1_full'], self.word2vec)
                    c1_elmo = self.fulltrainelmo.get(str(self.fulltrainidx[c1id]))
                    c1_emb = [np.concatenate([c1_w2v[i], c1_elmo[i]]) for i in range(len(c1_w2v))]

                    c2_w2v = features.encode(trainrow['q2_full'], self.word2vec)
                    c2_elmo = self.fulltrainelmo.get(str(self.fulltrainidx[c2id]))
                    c2_emb = [np.concatenate([c2_w2v[i], c2_elmo[i]]) for i in range(len(c2_w2v))]

                    c1_tree, c2_tree = treekernel.similar_terminals(c1_tree, c2_tree)

                    kq1 = self.memoize(q1id, q1_tree, q1_emb, q1id, q1_tree, q1_emb, treekernel)
                    kc1 = self.memoize(c1id, c1_tree, c1_emb, c1id, c1_tree, c1_emb, treekernel)
                    kq1c1 = float(self.memoize(q1id, q1_tree, q1_emb, c1id, c1_tree, c1_emb, treekernel)) / np.sqrt(kq1 * kc1) # normalized

                    kq2 = self.memoize(q2id, q2_tree, q2_emb, q2id, q2_tree, q2_emb, treekernel)
                    kc2 = self.memoize(c2id, c2_tree, c2_emb, c2id, c2_tree, c2_emb, treekernel)
                    kq2c2 = float(self.memoize(q2id, q2_tree, q2_emb, c2id, c2_tree, c2_emb, treekernel)) / np.sqrt(kq2 * kc2) # normalized

                    # kq1c2 = float(self.memoize(q1id, q1_tree, q1_emb, c2id, c2_tree, c2_emb, treekernel)) / np.sqrt(kq1 * kc2) # normalized
                    # kq2c1 = float(self.memoize(q2id, q2_tree, q2_emb, c1id, c1_tree, c1_emb, treekernel)) / np.sqrt(kq2 * kc1) # normalized

                    k = kq1c1 + kq2c2
                    X.append(k)
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


if __name__ == '__main__':
    logging.info('Load corpus', extra=d)
    trainset, devset = load.run()

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # Linear SVM
    linear = LinearSVM()
    linear.train()
    # # BM25
    # bm25 = BM25()
    # bm25.train()
    # # Tree SVM
    # semeval = TreeSVM()
    # semeval.train()
    # p.dump(semeval.memoization, open('data/treememoization.pickle', 'wb'))

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

    # # BM25 performance
    # devgold = utils.prepare_gold(GOLD_PATH)
    # bm25_ranking = bm25.validate()
    # map_baseline, map_model = utils.evaluate(devgold, copy.copy(bm25_ranking))
    #
    # print('Evaluation BM25')
    # print('MAP baseline: ', map_baseline)
    # print('MAP model: ', map_model)
    # print(10 * '-')

    # Tree performance
    # tree_ranking, y_real, y_pred = semeval.validate()
    # devgold = utils.prepare_gold(GOLD_PATH)
    # map_baseline, map_model = utils.evaluate(devgold, copy.copy(tree_ranking))
    # f1score = f1_score(y_real, y_pred)
    # accuracy = accuracy_score(y_real, y_pred)
    # print('Evaluation Tree')
    # print('MAP baseline: ', map_baseline)
    # print('MAP model: ', map_model)
    # print('Accuracy: ', accuracy)
    # print('F-Score: ', f1score)
    # print(10 * '-')

    # ranking = {}
    # for qid in linear_ranking:
    #     lrank = linear_ranking[qid]
    #     trank = tree_ranking[qid]
    #
    #     ranking[qid] = []
    #     for row1 in lrank:
    #         for row2 in trank:
    #             if row1[2] == row2[2]:
    #                 score = (0.8 * row1[1]) + (0.2 * row2[1])
    #                 ranking[qid].append((row1[0], score, row1[2]))
    #
    #
    # devgold = utils.prepare_gold(GOLD_PATH)
    # map_baseline, map_model = utils.evaluate(devgold, copy.copy(ranking))
    #
    # print('Evaluation Linear+Tree Weighted')
    # print('MAP baseline: ', map_baseline)
    # print('MAP model: ', map_model)
    # print(10 * '-')
    #
    #
    # ranking = {}
    # for qid in linear_ranking:
    #     lrank = linear_ranking[qid]
    #     trank = tree_ranking[qid]
    #
    #     ranking[qid] = []
    #     for row1 in lrank:
    #         for row2 in trank:
    #             if row1[2] == row2[2]:
    #                 score = row1[1] + row2[1]
    #                 ranking[qid].append((row1[0], score, row1[2]))
    #
    #
    # devgold = utils.prepare_gold(GOLD_PATH)
    # map_baseline, map_model = utils.evaluate(devgold, copy.copy(ranking))
    #
    # print('Evaluation Linear+Tree')
    # print('MAP baseline: ', map_baseline)
    # print('MAP model: ', map_model)
    # print(10 * '-')
