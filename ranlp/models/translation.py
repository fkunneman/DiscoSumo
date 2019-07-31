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
        lmprob, trmprob, trlmprob = 0.0, 0.0, 0.0
        if len(q1) == 0 or len(q2) == 0: return lmprob, trmprob, trlmprob, 0
        Q = pd.Series(q2)
        Q_count = Q.count()

        start = time.time()
        t_Qs = []
        for t in q2:
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(q1):
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
        lmprob, trmprob, trlmprob = 0.0, 0.0, 0.0
        if len(q1) == 0 or len(q2) == 0: return lmprob, trmprob, trlmprob, 0
        Q = pd.Series(q2)
        Q_count = Q.count()

        start = time.time()
        t_Qs = []
        for t in q2:
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(q1):
            w_C = self.get_w_C(w)

            ml_w_Q = float(Q[Q == w].count()) / Q_count
            mx_w_Q = 0.0

            for j, t in enumerate(q2):
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