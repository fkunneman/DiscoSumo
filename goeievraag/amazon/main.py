__author__='thiagocastroferreira'

import _pickle as p

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

import json
import os
import string
punctuation = string.punctuation
import pandas as pd
import numpy as np
import nltk.tokenize as tok

from gensim.summarization import bm25
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from sklearn.metrics.pairwise import cosine_similarity

# DATA_PATH='/roaming/fkunnema/goeievraag/data/'
DATA_PATH='data/'
SEEDS_PATH=os.path.join(DATA_PATH, 'seeds.json')

NEW_ANSWERS=os.path.join(DATA_PATH, 'answer_parsed_proc.json')

STOPWORDS_PATH=os.path.join(DATA_PATH, 'stopwords.txt')

NEW_TRAINDATA=os.path.join(DATA_PATH, 'ranked_questions_labeled_proc.json')

CORPUS_PATH=os.path.join(DATA_PATH, 'corpus.json')

DICT_PATH=os.path.join(DATA_PATH, 'dict.model')

# softcosine path
TFIDF_PATH=os.path.join(DATA_PATH, 'tfidf.model')
# translation path
TRANSLATION_PATH=os.path.join(DATA_PATH, 'translation.json')
# ensemble path
ENSEMBLE_PATH=os.path.join(DATA_PATH, 'ensemble.pkl')

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


class GoeieVraag():
    def __init__(self, w2v_dim=300, w2v_window=10, evaluation=False):
        self.w2v_dim = w2v_dim
        self.w2v_window = w2v_window
        self.evaluation = evaluation
        self.tokenizer = tok.toktok.ToktokTokenizer()

        self.load_datasets()

        # bm25
        self.init_bm25(self.seeds)
        # word2vec
        self.load_word2vec()
        # translation
        self.load_translation()
        # softcosine
        self.load_sofcos()
        # ensemble
        self.init_ensemble()


    def load_datasets(self):
        # STOPWORDS
        with open(STOPWORDS_PATH) as f:
            self.stopwords = [word.lower().strip() for word in f.read().split()]

        # QUESTIONS
        self.seeds = json.load(open(SEEDS_PATH))
        self.seed2idx = {q['id']:i for i, q in enumerate(self.seeds)}


        # DICTIONARIES
        self.dict = Dictionary.load(DICT_PATH)

        # PROCDATA
        procdata = json.load(open(NEW_TRAINDATA))
        self.procdata = procdata['procdata']
        self.traindata, self.testdata = procdata['train'], procdata['test']


    # BM25
    def init_bm25(self, corpus):
        ids, questions = [], []
        for row in corpus:
            ids.append(row['id'])
            questions.append(row['tokens'])


        self.idx2id = dict([(i, qid) for i, qid in enumerate(ids)])
        self.id2idx = dict([(qid, i) for i, qid in enumerate(ids)])
        self.bm25 = bm25.BM25(questions)


    def retrieve(self, query, n=30):
        scores = self.bm25.get_scores(query)
        questions = []
        for i in range(len(self.seeds)):
            questions.append({
                'id': self.seeds[i]['id'],
                'tokens': self.seeds[i]['tokens'],
                'text': self.seeds[i]['text'],
                'category': self.seeds[i]['category'],
                'score': scores[i]
            })

        return sorted(questions, key=lambda x: x['score'], reverse=True)[:n]


    # WORD2VEC
    def load_word2vec(self):
        fname = 'word2vec.' + str(self.w2v_dim) + '_' + str(self.w2v_window) + '.model'
        path = os.path.join(DATA_PATH, fname)
        self.word2vec = Word2Vec.load(path)


    def encode(self, question):
        emb = []
        for w in question:
            try:
                emb.append(self.word2vec[w.lower()])
            except:
                emb.append(self.w2v_dim * [0])
        return emb


    # Preprocessing
    def preprocess(self, q1, q2):
        if type(q1) == str:
            q1 = q1.split()
        if type(q2) == str:
            q2 = q2.split()

        q1emb = self.encode(q1)
        q2emb = self.encode(q2)

        return q1, q1emb, q2, q2emb


    # Softcosine
    def load_sofcos(self):
        self.tfidf = TfidfModel.load(TFIDF_PATH)


    def softcos(self, q1, q1emb, q2, q2emb):
        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    if w1[0] == w2[0]:
                        cos += (w1[1] * w2[1])
                    else:
                        m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                        cos += (w1[1] * m_ij * w2[1])
            return cos

        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = np.sqrt(dot(q1tfidf, q1emb, q1tfidf, q1emb))
        q2q2 = np.sqrt(dot(q2tfidf, q2emb, q2tfidf, q2emb))
        sofcosine = dot(q1tfidf, q1emb, q2tfidf, q2emb) / (q1q1 * q2q2)
        return sofcosine


    # Translation
    def init_translation(self, corpus, alpha, sigma):
        tokens = []
        for question in list(corpus):
            for token in question:
                tokens.append(token)

        Q_len = float(len(tokens))
        aux_w_Q = self.dict.doc2bow(tokens)
        aux_w_Q = dict([(self.dict[w[0]], (w[1]+1.0)/(Q_len+len(self.dict))) for w in aux_w_Q])

        w_Q = {}
        for w in aux_w_Q:
            if w[0] not in w_Q:
                w_Q[w[0]] = {}
            w_Q[w[0]][w] = aux_w_Q[w]

        self.alpha = alpha
        self.sigma = sigma
        self.prob_w_C = w_Q


    def load_translation(self):
        translation = json.load(open(TRANSLATION_PATH))
        self.prob_w_C = translation['w_Q']
        self.alpha = translation['alpha']
        self.sigma = translation['sigma']


    def translate(self, q1, q1emb, q2, q2emb):
        score = 0.0
        if len(q1) == 0 or len(q2) == 0: return 0.0

        Q = pd.Series(q2)
        Q_count = Q.count()

        t_Qs = []
        for t in q2:
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(q1):
            try:
                w_C = self.prob_w_C[w[0]][w]
            except:
                w_C = 1.0 / len(self.dict)

            ml_w_Q = float(Q[Q == w].count()) / Q_count
            mx_w_Q = 0.0

            for j, t in enumerate(q2):
                w_t = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0]) ** 2

                t_Q = t_Qs[j]
                mx_w_Q += (w_t * t_Q)
            w_Q = (self.sigma * mx_w_Q) + ((1-self.sigma) * ml_w_Q)
            score += np.log(((1-self.alpha) * w_Q) + (self.alpha * w_C))

        return score


    # Ensemble
    def init_ensemble(self):
        self.ensemble = Model()

        traindata = self.traindata if self.evaluation else self.procdata
        ids, questions = [], []
        for row in self.seeds:
            ids.append(row['id'])
            questions.append(row['tokens'])
        for q1id in traindata:
            for q2id in traindata[q1id]:
                if q1id not in ids:
                    ids.append(q1id)
                    questions.append(traindata[q1id][q2id]['q1'])
                if q2id not in ids:
                    ids.append(q2id)
                    questions.append(traindata[q1id][q2id]['q2'])

        id2idx = dict([(qid, i) for i, qid in enumerate(ids)])
        bm25_ = bm25.BM25(questions)

        X, y = [], []
        for q1id in traindata:
            for q2id in traindata[q1id]:
                pair = traindata[q1id][q2id]
                q1 = pair['q1']
                q2 = pair['q2']
                label = pair['label']

                q1, q1emb, q2, q2emb = self.preprocess(q1, q2)

                # TCF: ATTENTION ON THE ID HERE. WE NEED TO CHECK THIS
                bm25score = bm25_.get_score(q1, id2idx[q2id])
                translation = self.translate(q1, q1emb, q2, q2emb)
                softcosine = self.softcos(q1, q1emb, q2, q2emb)

                X.append([bm25score, translation, softcosine])
                y.append(label)

        self.ensemble.train_scaler(X)
        X = self.ensemble.scale(X)
        self.ensemble.train_regression(trainvectors=X, labels=y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def load_ensemble(self):
        if not os.path.exists(ENSEMBLE_PATH):
            self.init_ensemble()
        else:
            self.ensemble = Model()
            self.ensemble.load(ENSEMBLE_PATH)


    def ensembling(self, q1, q1emb, q2id, q2, q2emb):
        bm25score = self.bm25.get_score(q1, self.id2idx[q2id])
        translation = self.translate(q1, q1emb, q2, q2emb)
        softcosine = self.softcos(q1, q1emb, q2, q2emb)

        X = [bm25score, translation, softcosine]
        X = self.ensemble.scale([X])[0]
        clfscore, pred_label = self.ensemble.score(X)
        return clfscore, pred_label


    def __call__(self, query, method='ensemble'):
        # tokenize and lowercase
        tokens = [w.lower() for w in self.tokenizer.tokenize(query)]
        # remove stop words and punctuation
        query_proc = [w for w in tokens if w not in self.stopwords and w not in punctuation]

        # retrieve 30 candidates with bm25
        questions = self.retrieve(query_proc)
        # reranking with chosen method
        if method != 'bm25':
            questions = self.rerank(query=query_proc, questions=questions, method=method)

        result = { 'query': query, 'questions': [] }
        for question in questions:
            question_id = question['id']
            q = self.seeds[self.seed2idx[question_id]]
            q['score'] = question['score'] if method == 'bm25' else question['rescore']
            result['questions'].append(q)
        return result


    def rerank(self, query, questions, n=10, method='ensemble'):
        for i, question in enumerate(questions):
            q1, q1emb, q2, q2emb = self.preprocess(query, question['tokens'])

            if method == 'softcosine':
                questions[i]['rescore'] = self.softcos(q1, q1emb, q2, q2emb)
            elif method == 'translation':
                questions[i]['rescore'] = self.translate(q1, q1emb, q2, q2emb)
            else:
                questions[i]['rescore'], _ = self.ensembling(q1, q1emb, question['id'], q2, q2emb)

        questions = sorted(questions, key=lambda x: x['rescore'], reverse=True)[:n]
        return questions