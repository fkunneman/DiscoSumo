import csv
import os
from gensim import corpora
import pandas as pd
import numpy as np

import time

from multiprocessing import Pool

def load_stopwords():
    path = '/roaming/fkunnema/goeievraag/exp_similarity/stopwords.txt'
    with open(path) as f:
        stopwords = [word.lower().strip() for word in f.read().split()]
    return stopwords

def load_corpus(stopwords):
    goeievraag_path = '/roaming/fkunnema/goeievraag'  # goeievraag path
    # loading tokenized and 'stopworded' questions, and lowercase them.
    train_question_path = os.path.join(goeievraag_path, 'exp_similarity', 'train_questions.tok.txt')
    with open(train_question_path) as f:
        doc = f.read()

    original_train_questions, train_questions = [], []
    for question in doc.split('\n')[:-1]:
        original_question = [word for word in question.lower().split()]
        question = [word for word in original_question if word not in stopwords]
        if len(question) > 0:
            original_train_questions.append(original_question)
            train_questions.append(question)
    print('Number of train questions: ', str(len(train_questions)))

    path = os.path.join(goeievraag_path, 'exp_similarity', 'seed_questions.tok.txt')
    with open(path) as f:
        doc = f.read()

    original_test_questions, test_questions = [], []
    for question in doc.split('\n')[:-1]:
        original_question = [word for word in question.lower().split()]
        question = [word for word in original_question if word not in stopwords]
        if len(question) > 0:
            original_test_questions.append(original_question)
            test_questions.append(question)
    print('Number of test questions: ', str(len(test_questions)))

    return train_questions, test_questions, original_train_questions, original_test_questions

def get_w_C(train_questions, vocabulary):
    tokens = []
    for question in train_questions:
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


def translation_prob():
    path = '/home/tcastrof/DiscoSumo/translation/model/lex.f2e'
    with open(path) as f:
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


class Base():
    def __init__(self, training, prob_w_C, vocablen, alpha):
        self.training = training
        self.prob_w_C = prob_w_C
        self.vocablen = vocablen
        self.alpha = alpha

    def score(self, query, question, w_Cs=[]):
        raise NotImplementedError("Please Implement this method")

    def rank(self, query, n=10):
        ranking = []

        w_Cs = []
        for w in query:
            w = w.lower()
            try:
                w_C = self.prob_w_C[w[0]][w]
            except:
                w_C = 1.0 / self.vocablen
            w_Cs.append(w_C)

        for i, question in enumerate(self.training):
            if i % 1000 == 0:
                print('Train question: ', str(i), sep="\t")
            prob = self.score(query, question, w_Cs)
            ranking.append((question, prob))

        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        return ranking[:n]


class TRLM(Base):
    def __init__(self, training, prob_w_C, prob_w_t, vocablen, alpha, sigma):
        Base.__init__(self, training, prob_w_C, vocablen, alpha)
        self.prob_w_t = prob_w_t
        self.sigma = sigma

    # optimized
    def score_all(self, query, question, w_Cs=[]):
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
                try:
                    w_C = self.prob_w_C[w[0]][w]
                except:
                    w_C = 1.0 / self.vocablen
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

    def rank_all(self, query, n=10):
        lmrank, trmrank, trlmrank = [], [], []
        score_times = []

        w_Cs = []
        for w in query:
            w = w.lower()
            try:
                w_C = self.prob_w_C[w[0]][w]
            except:
                w_C = 1.0 / self.vocablen
            w_Cs.append(w_C)

        for i, question in enumerate(self.training):
            lmprob, trmprob, trlmprob, score_time  = self.score_all(query, question, w_Cs)
            score_times.append(score_time)
            lmrank.append((i, lmprob))
            trmrank.append((i, trmprob))
            trlmrank.append((i, trlmprob))
            if i % 1000 == 0:
                print('Train question: ', str(i), 'Score time: ', str(np.mean(score_times)), 'Total time: ', str(sum(score_times)), sep="\t")

        lmrank = sorted(lmrank, key=lambda x: x[1], reverse=True)
        trmrank = sorted(trmrank, key=lambda x: x[1], reverse=True)
        trlmrank = sorted(trlmrank, key=lambda x: x[1], reverse=True)
        return lmrank[:n], trmrank[:n], trlmrank[:n]


def run(original_test_questions, test_questions, train_questions, w_C, t2w, voclen, alpha, sigma):
    print('Load language model')
    trlm = TRLM(train_questions, w_C, t2w, voclen, alpha, sigma)  # translation-based language model

    lmranking, trmranking, trlmranking = [], [], []
    for i, q in enumerate(test_questions):
        lmrank, trmrank, trlmrank = trlm.rank_all(q)

        original_test_question = original_test_questions[i]
        print('Ranking: ', ' '.join(original_test_question))
        for rank in lmrank:
            idx_rank_question = rank[0]
            prob = rank[1]
            lmranking.append((original_test_question, idx_rank_question, prob))
        for rank in trmrank:
            idx_rank_question = rank[0]
            prob = rank[1]
            trmranking.append((original_test_question, idx_rank_question, prob))
        for rank in trlmrank:
            idx_rank_question = rank[0]
            prob = rank[1]
            trlmranking.append((original_test_question, idx_rank_question, prob))
    return lmranking, trmranking, trlmranking

if __name__ == '__main__':
    print('Load stopwords')
    stopwords = load_stopwords()
    print('Load corpus')
    train_questions, test_questions, original_train_questions, original_test_questions = load_corpus(stopwords)
    print('Load dictionary')
    vocabulary = corpora.Dictionary(train_questions)
    print('Load background probabilities')
    w_C = get_w_C(train_questions, vocabulary)  # background lm
    print('Load translation probabilities')
    t2w = translation_prob()  # translation probabilities

    THREADS = 25
    pool = Pool(processes=THREADS)
    n = int(len(test_questions) / THREADS)
    test_chunks = [(original_test_questions[i:i+n], test_questions[i:i+n]) for i in range(0, len(test_questions), n)]

    processes = []
    for chunk in test_chunks:
        processes.append(pool.apply_async(run, [chunk[0], chunk[1], train_questions, w_C, t2w, len(vocabulary), 0.5, 0.3]))

    lm, trm, trlm = [], [], []
    for process in processes:
        lmranking, trmranking, trlmranking = process.get()
        lm.extend(lmranking)
        trm.extend(trmranking)
        trlm.extend(trlmranking)

    print('Saving rankings...')
    with open('lmranking.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for result in lm:
            writer.writerow([' '.join(result[0]), ' '.join(original_train_questions[result[1]]), result[2]])
    with open('trmranking.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for result in trm:
            writer.writerow([' '.join(result[0]), ' '.join(original_train_questions[result[1]]), result[2]])
    with open('trlmranking.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for result in trlm:
            writer.writerow([' '.join(result[0]), ' '.join(original_train_questions[result[1]]), result[2]])