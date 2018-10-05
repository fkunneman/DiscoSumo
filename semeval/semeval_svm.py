__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import _pickle as p
import copy
import features
import json
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import re
import utils

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import f1_score

from translation import *

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'
FEATURE_PATH='trainfeatures.pickle'
MODEL_PATH='svm.model'

TRAIN_PATH='trainset.data'
DEV_PATH='devset.data'

TRANSLATION_PATH='translation/model/lex.f2e'

class SemevalSVM():
    def __init__(self, trainset, devset, testset):
        props={'annotators': 'tokenize,ssplit,pos,parse','pipelineLanguage':'en','outputFormat':'json'}
        corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

        print('\nPreparing development set...')
        if not os.path.exists(DEV_PATH):
            self.devset = utils.prepare_corpus(devset, corenlp=corenlp, props=props)
            json.dump(self.devset, open(DEV_PATH, 'w'))
        else:
            self.devset = json.load(open(DEV_PATH))
        self.devgold = utils.prepare_gold(GOLD_PATH)
        print('Preparing trainset...')
        if not os.path.exists(TRAIN_PATH):
            self.trainset = utils.prepare_corpus(trainset, corenlp=corenlp, props=props)
            json.dump(self.trainset, open(TRAIN_PATH, 'w'))
        else:
            self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.trainset)
        print('TRAIN DATA SIZE: ', len(self.traindata))
        print('\nPreparing test set...')
        self.testset = utils.prepare_corpus(testset, corenlp=corenlp, props=props)

        self.init_translation()

    def init_translation(self):
        print('\nLoad background probabilities')

        # TO DO: improve this
        questions = {}
        for trainrow in self.traindata:
            qid, q = trainrow['q1_id'], trainrow['q1']
            if qid not in questions:
                questions[qid] = q

            qid, q = trainrow['q2_id'], trainrow['q2']
            if qid not in questions:
                questions[qid] = q
        w_C = compute_w_C(questions, self.vocabulary)  # background lm
        print('Load translation probabilities')
        t2w = translation_prob(TRANSLATION_PATH)  # translation probabilities
        self.translation = TRLM([], w_C, t2w, len(self.vocabulary), 0.6, 0.6)  # translation-based language model

    def __transform__(self, q1, q2):
        if type(q1) == list: q1 = ' '.join(q1)
        if type(q2) == list: q2 = ' '.join(q2)

        lcs = len(features.lcs(re.split('(\W)', q1), re.split('(\W)', q2))[1].split())
        lcsub = features.lcsub(q1, q2)[0]
        jaccard = features.jaccard(q1, q2)
        containment_similarity = features.containment_similarities(q1, q2)
        lmprob, trmprob, trlmprob, proctime = self.translation.score(q1.split(), q2.split())

        X = [lcs, lcsub, jaccard, containment_similarity, lmprob, trmprob]

        # ngram features
        for n in range(2, 5):
            ngram1 = ' '
            for gram in nltk.ngrams(q1.split(), n):
                ngram1 += 'x'.join(gram) + ' '

            ngram2 = ' '
            for gram in nltk.ngrams(q2.split(), n):
                ngram2 += 'x'.join(gram) + ' '

            X.append(len(features.lcs(re.split('(\W)', ngram1), re.split('(\W)', ngram2))[1].split()))
            X.append(features.lcsub(ngram1, ngram2)[0])
            X.append(features.jaccard(ngram1, ngram2))
            X.append(features.containment_similarities(ngram1, ngram2))

        return X


    def train(self):
        treekernel = features.TreeKernel()

        if not os.path.exists(FEATURE_PATH):
            X, y = [], []
            for i, query_question in enumerate(self.traindata):
                percentage = round(float(i+1) / len(self.traindata), 2)
                print('Preparing traindata: ', percentage, sep='\t', end='\r')
                q1, q2 = query_question['q1'], query_question['q2']
                x = self.__transform__(q1, q2)

                q1, q2 = query_question['q1_tree'], query_question['q2_tree']
                x.append(treekernel(q1, q2))

                X.append(x)
                y.append(query_question['label'])

            p.dump(list(zip(X, y)), open(FEATURE_PATH, 'wb'))
        else:
            f = p.load(open(FEATURE_PATH, 'rb'))
            X = list(map(lambda x: x[0], f))
            y = list(map(lambda x: x[1], f))

        if not os.path.exists(MODEL_PATH):
            self.train_classifier(
                trainvectors=X,
                labels=y,
                c='search',
                kernel='linear',
                gamma='search',
                degree='search',
                jobs=10
            )
        else:
            self.model = p.load(open(MODEL_PATH, 'rb'))


    def train_classifier(self, trainvectors, labels, c='1.0', kernel='linear', gamma='0.1', degree='1', class_weight='balanced', iterations=10, jobs=1):
        parameters = ['C', 'kernel', 'gamma', 'degree']
        if len(class_weight.split(':')) > 1: # dictionary
            class_weight = dict([label_weight.split(':') for label_weight in class_weight.split()])
        c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        kernel_values = ['linear', 'rbf', 'poly'] if kernel == 'search' else [k for  k in kernel.split()]
        gamma_values = [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048] if gamma == 'search' else [float(x) for x in gamma.split()]
        degree_values = [1, 2, 3, 4] if degree == 'search' else [int(x) for x in degree.split()]
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
        self.model = svm.SVC(
            probability = True,
            C = settings[parameters[0]],
            kernel = settings[parameters[1]],
            gamma = settings[parameters[2]],
            degree = settings[parameters[3]],
            class_weight = class_weight,
            cache_size = 1000,
            verbose = 2
        )
        self.model.fit(trainvectors, labels)
        p.dump(self.model, open(MODEL_PATH, 'wb'))


    def validate(self):
        treekernel = features.TreeKernel()
        ranking = {}
        y_real, y_pred = [], []
        for i, qid in enumerate(self.devset):
            ranking[qid] = []
            percentage = round(float(i+1) / len(self.devset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')

            query = self.devset[qid]
            q1 = query['tokens']
            q1_tree = query['tree']

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                rel_question_id = rel_question['id']

                q2 = rel_question['tokens']
                q2_tree = rel_question['tree']
                X = self.__transform__(q1, q2)
                X.append(treekernel(q1_tree, q2_tree))

                score = self.model.predict_proba([X])[0][1]
                if score > 0.5:
                    pred_label = 1
                else:
                    pred_label = 0
                y_pred.append(pred_label)

                if rel_question['relevance'] != 'Irrelevant':
                    y_real.append(1)
                else:
                    y_real.append(0)
                ranking[qid].append((pred_label, score, rel_question_id))

        map_baseline, map_model = utils.evaluate(self.devgold, ranking)
        f1score = f1_score(y_real, y_pred)
        return map_baseline, map_model, f1score


if __name__ == '__main__':
    print('Load corpus')
    trainset, devset = load.run()

    semeval = SemevalSVM(trainset, devset, [])
    semeval.train()

    map_baseline, map_model, f1score = semeval.validate()

    print('Evaluation')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('F-Score: ', f1score)