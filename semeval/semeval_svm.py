__author__='thiagocastroferreira'

import cPickle as p
import copy
import features
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import re
import utils

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import f1_score

GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

class SemevalSVM():
    def __init__(self, trainset, devset, testset):
        print('Preparing trainset...')
        self.trainset = utils.prepare_corpus(trainset)
        self.traindata, self.voc2id, self.id2voc = utils.prepare_traindata(self.trainset)
        print('TRAIN DATA SIZE: ', len(self.traindata))
        print('\nPreparing development set...')
        self.devset = utils.prepare_corpus(devset)
        self.devgold = utils.prepare_gold(GOLD_PATH)
        print('\nPreparing test set...')
        self.testset = utils.prepare_corpus(testset)


    def __transform__(self, q1, q2, treekernel):
        if not treekernel:
            treekernel = features.TreeKernel()
        if type(q1) == list: q1 = ' '.join(q1)
        if type(q2) == list: q2 = ' '.join(q2)

        treek = treekernel(q1, q2)
        lcs = features.lcs(re.split('(\W)', q1), re.split('(\W)', q2))[0]
        lcsub = features.lcsub(q1, q2)[0]
        jaccard = features.jaccard(q1, q2)
        containment_similarity = features.containment_similarities(q1, q2)

        X = [treek, lcs, lcsub, jaccard, containment_similarity]

        # ngram features
        for n in range(2, 5):
            ngram1 = ''
            for gram in nltk.ngrams(q1.split(), n):
                ngram1 += 'x'.join(gram) + ' '

            ngram2 = ''
            for gram in nltk.ngrams(q2.split(), n):
                ngram2 += 'x'.join(gram) + ' '

            X.append(features.lcs(re.split('(\W)', ngram1), re.split('(\W)', ngram2))[0])
            X.append(features.lcs(ngram1, ngram2)[0])
            X.append(features.jaccard(ngram1, ngram2))
            X.append(features.containment_similarities(ngram1, ngram2))

        return X


    def train(self):
        treekernel = features.TreeKernel()

        X, y = [], []
        for i, query_question in enumerate(self.traindata):
            percentage = round(float(i+1) / len(self.traindata), 2)
            print('Preparing traindata: ', percentage, sep='\t', end='\r')
            q1, q2 = query_question['q1'], query_question['q2']
            x = self.__transform__(q1, q2, treekernel)
            X.append(x)
            y.append(query_question['label'])



        self.train_classifier(
            trainvectors=X,
            labels=y,
            c='search',
            kernel='linear',
            gamma='search',
            degree='search'
        )


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
        p.dump(self.model, open('svm.model', 'w'))


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

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                rel_question_id = rel_question['id']

                q2 = rel_question['tokens']
                X = self.__transform__(q1, q2, treekernel)
                score = self.model.predict_proba(X)[0][1]
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

        gold = copy.copy(self.devset)
        map_baseline, map_model = utils.evaluate(gold, ranking)
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