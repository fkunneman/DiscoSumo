__author__='thiagocastroferreira'

import features
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import re
import utils

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV

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

        lcs = features.lcs(re.split('(\W)', q1), re.split('(\W)', q2))[0]
        lcsub = features.lcsub(q1, q2)[0]
        jaccard = features.jaccard(q1, q2)
        treek = treekernel(q1, q2)
        return lcs, lcsub, jaccard, treek

    def train(self):
        treekernel = features.TreeKernel()

        X, y = [], []
        for query_question in self.traindata:
            q1, q2 = query_question['q1'], query_question['q2']
            x = self.__transform__(q1, q2, treekernel)
            X.append(x)
            y.append(query_question['label'])


        self.train_classifier(
            trainvectors=X,
            labels=y,
            c='search',
            kernel='search',
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
