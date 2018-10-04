__author__='thiagocastroferreira'

import numpy as np
import pandas as pd
import time

def compute_w_C(train_questions, vocabulary):
    tokens = []
    for question in list(train_questions.values()):
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


def translation_prob(TRANSLATION_PATH):
    with open(TRANSLATION_PATH) as f:
        doc = list(map(lambda x: x.split(), f.read().split('\n')))

    t2w = {}
    for row in doc[:-1]:
        t = row[0]
        if t[0] not in t2w:
            t2w[t[0]] = {}
        if t not in t2w[t[0]]:
            t2w[t[0]][t] = {}

        w = row[1]
        if w[0] not in t2w[t[0]][t]:
            t2w[t[0]][t][w[0]] = {}

        prob = float(row[2])
        t2w[t[0]][t][w[0]][w] = prob
    return t2w


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

    # optimized
    def score(self, query, question, w_Cs=[]):
        Q = pd.Series(question)
        Q_count = Q.count()
        lmprob, trmprob, trlmprob = 0.0, 0.0, 0.0

        start = time.time()
        t_Qs = []
        for t in question:
            t = t.lower()
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(query):
            w = w.lower()
            if len(w_Cs) == 0:
                w_C = self.get_w_C(w)
            else:
                w_C = w_Cs[i]

            ml_w_Q = float(Q[Q == w].count()) / Q_count
            mx_w_Q = 0.0

            for j, t in enumerate(question):
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
            lmprob, trmprob, trlmprob, score_time  = self.score(query, question, w_Cs)
            score_times.append(score_time)
            lmrank.append((question_id, lmprob))
            trmrank.append((question_id, trmprob))
            trlmrank.append((question_id, trlmprob))

        lmrank = list(map(lambda x: x[0], sorted(lmrank, key=lambda x: x[1], reverse=True)))
        trmrank = list(map(lambda x: x[0], sorted(trmrank, key=lambda x: x[1], reverse=True)))
        trlmrank = list(map(lambda x: x[0], sorted(trlmrank, key=lambda x: x[1], reverse=True)))
        return lmrank[:n], trmrank[:n], trlmrank[:n]