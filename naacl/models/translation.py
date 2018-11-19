__author__='thiagocastroferreira'

import numpy as np
import pandas as pd
import time

from sklearn.metrics.pairwise import cosine_similarity

def compute_w_C(corpus, vocabulary):
    tokens = []
    for question in list(corpus):
        for token in question:
            tokens.append(token)

    vocablen = len(vocabulary)
    Q_len = float(len(tokens))
    aux_w_Q = vocabulary.doc2bow(tokens)
    aux_w_Q = dict([(vocabulary[w[0]], (w[1]+1.0)/(Q_len+vocablen)) for w in aux_w_Q])

    w_Q = {}
    for w in aux_w_Q:
        if w[0] not in w_Q:
            w_Q[w[0]] = {}
        w_Q[w[0]][w] = aux_w_Q[w]
    return w_Q


class TRLM():
    def __init__(self, training, prob_w_C, prob_w_t, vocablen, alpha, sigma):
        self.training = training
        self.prob_w_C = prob_w_C
        self.vocablen = vocablen
        self.alpha = alpha
        self.prob_w_t = prob_w_t
        self.sigma = sigma

    def get_w_C(self, w):
        try:
            w_C = self.prob_w_C[w[0]][w]
        except:
            w_C = 1.0 / self.vocablen
        return w_C

    def __call__(self, q1, q1_emb, q2, q2_emb):
        Q = pd.Series(q2)
        Q_count = Q.count()
        lmprob, trmprob, trlmprob = 0.0, 0.0, 0.0

        start = time.time()
        t_Qs = []
        for t in q2:
            t = t.lower()
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(q1):
            w = w.lower()
            w_C = self.get_w_C(w)

            ml_w_Q = float(Q[Q == w].count()) / Q_count
            mx_w_Q = 0.0

            for j, t in enumerate(q2):
                w_t = max(0, cosine_similarity([q1_emb[i]], [q2_emb[j]])[0][0]) ** 2

                t_Q = t_Qs[j]
                mx_w_Q += (w_t * t_Q)
            w_Q = (self.sigma * mx_w_Q) + ((1-self.sigma) * ml_w_Q)
            lmprob += np.log(((1-self.alpha) * ml_w_Q) + (self.alpha * w_C))
            trmprob += np.log(((1-self.alpha) * mx_w_Q) + (self.alpha * w_C))
            trlmprob += np.log(((1-self.alpha) * w_Q) + (self.alpha * w_C))
        end = time.time()
        return lmprob, trmprob, trlmprob, (end-start)


    def score(self, q1, q2):
        Q = pd.Series(q2)
        Q_count = Q.count()
        lmprob, trmprob, trlmprob = 0.0, 0.0, 0.0

        start = time.time()
        t_Qs = []
        for t in q2:
            t = t.lower()
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(q1):
            w = w.lower()
            w_C = self.get_w_C(w)

            ml_w_Q = float(Q[Q == w].count()) / Q_count
            mx_w_Q = 0.0

            for j, t in enumerate(q2):
                t = t.lower()
                try:
                    w_t = self.prob_w_t[t[0]][t][w[0]][w]
                except:
                    w_t = 0.0

                t_Q = t_Qs[j]
                mx_w_Q += (w_t * t_Q)
            w_Q = (self.sigma * mx_w_Q) + ((1-self.sigma) * ml_w_Q)
            lmprob += np.log(((1-self.alpha) * ml_w_Q) + (self.alpha * w_C))
            trmprob += np.log(((1-self.alpha) * mx_w_Q) + (self.alpha * w_C))
            trlmprob += np.log(((1-self.alpha) * w_Q) + (self.alpha * w_C))
        end = time.time()
        return lmprob, trmprob, trlmprob, (end-start)

    def rank(self, query, n=10):
        lmrank, trmrank, trlmrank = [], [], []
        score_times = []

        w_Cs = []
        for w in query:
            w = w.lower()
            w_C = self.get_w_C(w)
            w_Cs.append(w_C)

        for i, question_id in enumerate(self.training):
            question = self.training[question_id]
            lmprob, trmprob, trlmprob, score_time  = self.simple_score(query, question, w_Cs)
            score_times.append(score_time)
            lmrank.append((question_id, lmprob))
            trmrank.append((question_id, trmprob))
            trlmrank.append((question_id, trlmprob))

        lmrank = list(map(lambda x: x[0], sorted(lmrank, key=lambda x: x[1], reverse=True)))
        trmrank = list(map(lambda x: x[0], sorted(trmrank, key=lambda x: x[1], reverse=True)))
        trlmrank = list(map(lambda x: x[0], sorted(trlmrank, key=lambda x: x[1], reverse=True)))
        return lmrank[:n], trmrank[:n], trlmrank[:n]