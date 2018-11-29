__author='thiagocastroferreira'

import sys
sys.path.append('../')

import os

from semeval_svm import SemevalSVM
from semeval_kernel import SemevalTreeKernel

from models.svm import Model
from sklearn.preprocessing import MinMaxScaler

DATA_PATH='data'
FEATURE_PATH='feature'

class SemevalEnsemble:
    def __init__(self, classifiers, kernels, scale=True, path=''):
        self.path = path
        self.scale = scale
        self.classifiers = classifiers
        self.kernels = kernels

        self.ensemble = Model()
        self.train()


    def train_classifiers(self):
        for i , classifier in enumerate(self.classifiers):
            # model name
            model_name = classifier['model'] + '.' + classifier['path']
            self.features.append(model_name)

            print('Model: ', model_name)
            model = SemevalSVM(model=classifier['model'],
                               features=classifier['features'],
                               comment_features=classifier['comment_features'],
                               stop=classifier['stop'],
                               vector=classifier['vector'],
                               path=classifier['path'],
                               alpha=classifier['alpha'],
                               sigma=classifier['sigma'])

            if i == 0: self.y = model.y

            X = model.scaler(model.X)
            scores = model.svm.model.decision_function(X)
            if len(self.X) == 0:
                self.X = scores
            else:
                for j, score in enumerate(scores):
                    self.X[j].append(score)

            # model validation
            ranking, y_real, y_pred, parameter_settings = model.validate()
            for q1id in ranking:
                if q1id not in self.devrank:
                    self.devrank[q1id] = {}

                for question in ranking[q1id]:
                    label, score, q2id = question
                    if q2id not in self.devrank[q1id]:
                        self.devrank[q1id][q2id] = {}
                    self.devrank[q1id][q2id][model_name] = float(score)

            # testset 2016
            ranking, y_real, y_pred, parameter_settings = model.test(testset=model.testset2016,
                                                                     elmovec=model.test2016elmo, elmoidx=model.test2016idx,
                                                                     fullelmovec=model.fulltest2016elmo, fullelmoidx=model.fulltest2016idx)
            for q1id in ranking:
                if q1id not in self.test2016rank:
                    self.test2016rank[q1id] = {}

                for question in ranking[q1id]:
                    label, score, q2id = question
                    if q2id not in self.test2016rank[q1id]:
                        self.test2016rank[q1id][q2id] = {}
                    self.test2016rank[q1id][q2id][model_name] = float(score)

            # testset 2017
            ranking, y_real, y_pred, parameter_settings = model.test(testset=model.testset2017,
                                                                     elmovec=model.test2017elmo, elmoidx=model.test2017idx,
                                                                     fullelmovec=model.fulltest2017elmo, fullelmoidx=model.fulltest2017idx)
            for q1id in ranking:
                if q1id not in self.test2017rank:
                    self.test2017rank[q1id] = {}

                for question in ranking[q1id]:
                    label, score, q2id = question
                    if q2id not in self.test2017rank[q1id]:
                        self.test2017rank[q1id][q2id] = {}
                    self.test2017rank[q1id][q2id][model_name] = float(score)


    def train_kernels(self):
        for i , kernel in enumerate(self.kernels):
            # kernel name
            kernel_name = kernel['kernel_path']
            self.features.append(kernel_name)
            print('Kernel: ', kernel_name)
            model = SemevalTreeKernel(smoothed=kernel['smoothed'], vector=kernel['vector'], tree=kernel['tree'], kernel_path=kernel['kernel_path'])

            if i == 0: self.y = model.y
            for j, x in enumerate(list(model.X)):
                percentage = round(float(j + 1) / len(model.X), 2)
                print('Progress: ', percentage, j + 1, sep='\t', end = '\r')

                score, pred_label = model.svm.score(x)
                if len(model.X) != len(self.X):
                    self.X.append([score])
                else:
                    self.X[j].append(score)

            # model validation
            ranking, y_real, y_pred, parameter_settings = model.validate()
            self.devrank[kernel_name] = ranking

            # testset 2016
            ranking, y_real, y_pred, parameter_settings = model.test(testset=model.testset2016, elmovec=model.fulltest2016elmo, elmoidx=model.fulltest2016idx)
            self.test2016rank[kernel_name] = ranking

            # testset 2017
            ranking, y_real, y_pred, parameter_settings = model.test(testset=model.testset2017, elmovec=model.fulltest2017elmo, elmoidx=model.fulltest2017idx)
            self.test2017rank[kernel_name] = ranking


    def train(self):
        self.X, self.y = [], []
        self.features = []
        self.devrank, self.test2016rank, self.test2017rank = {}, {}, {}

        self.train_classifiers()
        self.train_kernels()

        if self.scale:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)
        self.ensemble.train_regression(trainvectors=self.X, labels=self.y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def test(self, set_='dev'):
        if set_ == 'dev':
            self.ranking = self.devrank
        elif set_ == 'test2016':
            self.ranking = self.test2016rank
        else:
            self.ranking = self.test2017rank

        ranking = {}
        y_real, y_pred = [], []
        for q1id in self.ranking:
            ranking[q1id] = []
            for q2id in self.ranking[q1id]:
                X = []
                for model in self.features:
                    X.append(self.ranking[q1id][q2id][model])

                if self.scale:
                    X = self.scaler.transform([X])[0]
                score, pred_label = self.ensemble.score(X)
                y_pred.append(pred_label)

                real_label = 'true'
                y_real.append(real_label)
                ranking[q1id].append((pred_label, score, q2id))

        parameter_settings = self.ensemble.return_parameter_settings(clf='regression')
        return ranking, y_real, y_pred, parameter_settings

def save(ranking, path, parameter_settings):
    with open(path, 'w') as f:
        f.write(parameter_settings)
        f.write('\n')
        for q1id in ranking:
            for row in ranking[q1id]:
                label = 'false'
                if row[0] == 1:
                    label = 'true'
                f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))

def evaluate(classifiers, kernels, scale, evaluation_path):
    ensemble = SemevalEnsemble(classifiers=classifiers, kernels=kernels, scale=scale)

    # DEV
    path = os.path.join(DEV_EVAL_PATH, evaluation_path)
    ranking, y_real, y_pred, parameter_settings = ensemble.test(set_='dev')
    save(ranking=ranking, path=path, parameter_settings=parameter_settings)

    # TEST2016
    path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)
    ranking, y_real, y_pred, parameter_settings = ensemble.test(set_='test2016')
    save(ranking=ranking, path=path, parameter_settings=parameter_settings)

    # TEST2017
    path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)
    ranking, y_real, y_pred, parameter_settings = ensemble.test(set_='test2017')
    save(ranking=ranking, path=path, parameter_settings=parameter_settings)


if __name__ == '__main__':
    EVALUATION_PATH='evaluation'
    DEV_EVAL_PATH=os.path.join(EVALUATION_PATH, 'dev')
    TEST2016_EVAL_PATH=os.path.join(EVALUATION_PATH, 'test2016')
    TEST2017_EVAL_PATH=os.path.join(EVALUATION_PATH, 'test2017')

    # Regression / Softcosine / word2vec+elmo + Regression / translation / word2vec + scaled
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.word2vec_elmo.features')
    classifier1 = {
        'model': 'regression',
        'features': 'softcosine', 'comment_features': 'softcosine',
        'stop': True, 'vector': 'word2vec+elmo', 'path': feature_path, 'alpha':0.7, 'sigma':0.3,
    }

    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.word2vec.features')
    classifier2 = {
        'model': 'regression',
        'features': 'translation', 'comment_features': 'translation',
        'stop': True, 'vector': 'word2vec', 'path': feature_path, 'alpha':0.7, 'sigma':0.3,
    }
    classifiers = [classifier1, classifier2]
    evaluation_path = 'ensemble.regression.softcosine.word2vec_elmo_regression.translation.word2vec.scaled'
    evaluate(classifiers=classifiers, kernels=[], evaluation_path=evaluation_path, scale=True)
    ###############################################################################

    # Regression / Softcosine / word2vec+elmo + Regression / translation / word2vec + notscaled
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.word2vec_elmo.features')
    classifier1 = {
        'model': 'regression',
        'features': 'softcosine', 'comment_features': 'softcosine',
        'stop': True, 'vector': 'word2vec+elmo', 'path': feature_path, 'alpha':0.7, 'sigma':0.3,
    }

    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.word2vec.features')
    classifier2 = {
        'model': 'regression',
        'features': 'translation', 'comment_features': 'translation',
        'stop': True, 'vector': 'word2vec', 'path': feature_path, 'alpha':0.7, 'sigma':0.3,
    }
    classifiers = [classifier1, classifier2]
    evaluation_path = 'ensemble.regression.softcosine.word2vec_elmo_regression.translation.word2vec.scaled'
    evaluate(classifiers=classifiers, kernels=[], evaluation_path=evaluation_path, scale=False)
    ###############################################################################


