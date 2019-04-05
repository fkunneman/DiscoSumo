__author__='thiagocastroferreira'

import _pickle as p

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

class Model():
    # Machine Learning
    def train_svm(self, trainvectors, labels, c='1.0', kernel='linear', gamma='0.1', degree='1', class_weight='balanced', iterations=10, jobs=1, gridsearch='random'):
        parameters = ['C', 'kernel', 'gamma', 'degree']
        c_values = [1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        kernel_values = ['linear', 'rbf', 'poly'] if kernel == 'search' else [k for  k in kernel.split()]
        gamma_values = ['auto','scale'] if gamma == 'search' else [float(x) for x in gamma.split()]
        degree_values = [1, 2, 3] if degree == 'search' else [int(x) for x in degree.split()]
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

            if gridsearch == 'random':
                paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            elif gridsearch == 'brutal':
                paramsearch = GridSearchCV(model, param_grid, cv = 5, verbose = 2, n_jobs = jobs, pre_dispatch = 4, refit = True)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_

        # train an SVC classifier with the settings that led to the best performance
        self.model = svm.SVC(
            probability = True,
            C = settings[parameters[0]],
            kernel = settings[parameters[1]],
            gamma = settings[parameters[2]],
            degree = settings[parameters[3]],
            cache_size = 1000,
            verbose = 2
        )
        self.model.fit(trainvectors, labels)


    def train_regression(self, trainvectors, labels, c='1.0', penalty='l1', tol='1e-4', solver='saga', iterations=10, jobs=1, gridsearch='random'):
        parameters = ['C', 'penalty']
        c_values = [1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        penalty_values = ['l1', 'l2'] if penalty == 'search' else [k for  k in penalty.split()]
        # tol_values = [1, 0.1, 0.01, 0.001, 0.0001] if tol == 'search' else [float(x) for x in tol.split()]
        grid_values = [c_values, penalty_values]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = LogisticRegression(solver=solver)

            if gridsearch == 'random':
                paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            elif gridsearch == 'brutal':
                paramsearch = GridSearchCV(model, param_grid, cv = 5, verbose = 2, n_jobs = jobs, pre_dispatch = 4, refit = True)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_
        # train an SVC classifier with the settings that led to the best performance
        self.model = LogisticRegression(
            C = settings[parameters[0]],
            penalty = settings[parameters[1]],
            # tol = settings[parameters[2]],
            solver= solver,
            verbose = 2
        )
        self.model.fit(trainvectors, labels)


    def score(self, X):
        score = self.model.decision_function([X])[0]
        pred_label = self.model.predict([X])[0]
        return score, pred_label


    def return_parameter_settings(self, clf='svm'):
        parameter_settings = []
        if clf == 'svm':
            params = ['C','kernel','gamma','degree']
        elif clf == 'regression':
            params = ['C', 'penalty', 'tol']
        else:
            params = []
        for param in params:
            parameter_settings.append([param,str(self.model.get_params()[param])])
        return ','.join([': '.join(x) for x in parameter_settings])


    def train_scaler(self, X):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(X)

    def scale(self, X):
        return self.scaler.transform(X)


    def save(self, path):
        model = { 'model': self.model, 'scaler': self.scaler }
        p.dump(model, open(path, 'wb'))


    def load(self, path):
        model = p.load(open(path, 'rb'))
        self.model = model['model']
        self.scaler = model['scaler']