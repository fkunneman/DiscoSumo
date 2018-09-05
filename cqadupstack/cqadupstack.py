__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/cqadupstack/CQADupStack')
import query_cqadupstack as qcqa

import os
from gensim import corpora
import pandas as pd
import numpy as np
import re
import time
from multiprocessing import Pool

CORPUS_PATH = '/home/tcastrof/Question/cqadupstack'
CATEGORY = 'android'
TRANSLATION_PATH=os.path.join('/home/tcastrof/Question/cqadupstack', CATEGORY, 'translation/model/lex.f2e')

QUESTION_TYPE='title'

def load_corpus():
    o = qcqa.load_subforum(os.path.join(CORPUS_PATH, CATEGORY + '.zip'))
    testset, develset, indexset = o.split_for_retrieval()
    return o, indexset, develset, testset

def remove_long_tokens(snt):
    _snt = []
    for word in snt.split():
        if len(word) <= 20:
            _snt.append(word)
    return ' '.join(_snt)

def prepare_questions(indexset):
    questions = {}
    vocabulary = []
    for i, idx in enumerate(indexset):
        try:
            print('Question number: ', i, 'Question index: ', idx, sep='\t', end='\r')
            # retrieve question
            if QUESTION_TYPE == 'title':
                q = o.get_posttitle(idx)
            else:
                q = o.get_post_title_and_body(idx)
            # removing stopwords and stemming it
            q = o.perform_cleaning(q, remove_stopwords=True, remove_punct=False, stem=True)
            # removing punctuation (better than in nltk)
            q = re.sub(r'[^\w\s][ ]*','', q).strip()
            # removing tokens greater than 20
            q = remove_long_tokens(q)

            q = q.split()
            if len(q) > 0:
                questions[idx] = q
                vocabulary.append(q)
        except:
            print('Question Error')
    vocabulary = corpora.Dictionary(vocabulary)
    return questions, vocabulary

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


def translation_prob():
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

            # if i % 500 == 0:
            #     score_time = round(np.mean(score_times), 2)
            #     total = round(sum(score_times), 2)
            #     percentage = str(round((float(i+1) / len(self.training))*100,2)) + '%'
            #     print('Question Number: ', i, 'Question ID: ', question_id, 'Training:', percentage, 'Score time: ', score_time, 'Total time: ', total, sep='\t', end='\r')

        lmrank = list(map(lambda x: x[0], sorted(lmrank, key=lambda x: x[1], reverse=True)))
        trmrank = list(map(lambda x: x[0], sorted(trmrank, key=lambda x: x[1], reverse=True)))
        trlmrank = list(map(lambda x: x[0], sorted(trlmrank, key=lambda x: x[1], reverse=True)))
        return lmrank[:n], trmrank[:n], trlmrank[:n]


def run(thread_id, o, testset, train_questions, w_C, t2w, voclen, alpha, sigma):
    print('Load language model')
    trlm = TRLM(train_questions, w_C, t2w, voclen, alpha, sigma)  # translation-based language model

    lmranking, trmranking, trlmranking = {}, {}, {}
    for i, idx in enumerate(testset):
        percentage = str(round((float(i+1) / len(testset))*100,2)) + '%'
        print('Thread ID:', thread_id, 'Query Number: ', i, 'Query ID: ', idx, 'Percentage:', percentage, sep='\t')
        try:
            if QUESTION_TYPE == 'title':
                q = o.get_posttitle(idx)
            else:
                q = o.get_post_title_and_body(idx)
            query = o.perform_cleaning(q, remove_stopwords=True, remove_punct=False, stem=True)
            # removing punctuation (better than in nltk)
            query = re.sub(r'[^\w\s][ ]*','', query).strip()
            # removing tokens greater than 20
            query = remove_long_tokens(query)
            query = query.split()

            lmrank, trmrank, trlmrank = trlm.rank(query)
        except:
            lmrank, trmrank, trlmrank = [], [], []

        lmranking[idx] = lmrank
        trmranking[idx] = trmrank
        trlmranking[idx] = trlmrank
    return lmranking, trmranking, trlmranking

if __name__ == '__main__':
    print('Load corpus')
    o, indexset, develset, testset = load_corpus()
    print('Preparing training questions and vocabulary')
    train_questions, vocabulary = prepare_questions(indexset)
    print('\nLoad background probabilities')
    w_C = compute_w_C(train_questions, vocabulary)  # background lm
    print('Load translation probabilities')
    t2w = translation_prob()  # translation probabilities

    THREADS = 25
    pool = Pool(processes=THREADS)
    n = int(len(develset) / THREADS)
    chunks = [develset[i:i+n] for i in range(0, len(develset), n)]

    processes = []
    for i, chunk in enumerate(chunks):
        processes.append(pool.apply_async(run, [i, o, chunk, train_questions, w_C, t2w, len(vocabulary), 0.5, 0.3]))

    lmranking, trmranking, trlmranking = {}, {}, {}
    for process in processes:
        lm, trm, trlm = process.get()
        lmranking.update(lm)
        trmranking.update(trm)
        trlmranking.update(trlm)

    # lmranking, trmranking, trlmranking = run(o, develset, train_questions, w_C, t2w, len(vocabulary), 0.5, 0.3)

    with open(os.path.join(CATEGORY, 'lmranking.txt'), 'w') as f:
        for query_id in lmranking:
            f.write(query_id)
            f.write(' ')
            f.write(' '.join(lmranking[query_id]))
            f.write(' <br />\n')

    with open(os.path.join(CATEGORY, 'trmranking.txt'), 'w') as f:
        for query_id in trmranking:
            f.write(query_id)
            f.write(' ')
            f.write(' '.join(trmranking[query_id]))
            f.write(' <br />\n')

    with open(os.path.join(CATEGORY, 'trlmranking.txt'), 'w') as f:
        for query_id in trlmranking:
            f.write(query_id)
            f.write(' ')
            f.write(' '.join(trlmranking[query_id]))
            f.write(' <br />\n')

    print('EVALUATION ', 'CATEGORY: ', CATEGORY, 'QUESTION TYPE: ', QUESTION_TYPE, sep='\t')
    print('Mean Average Precision (MAP)')
    path = os.path.join(CORPUS_PATH, CATEGORY, 'lmranking.txt')
    print('LM: ', o.mean_average_precision(path))
    print(10 * '-')
    path = os.path.join(CORPUS_PATH, CATEGORY, 'trmranking.txt')
    print('TRM: ', o.mean_average_precision(path))
    print(10 * '-')
    path = os.path.join(CORPUS_PATH, CATEGORY, 'trlmranking.txt')
    print('TRLM: ', o.mean_average_precision(path))
    print(10 * '-')

    print('\n\n')

    print('Precision')
    path = os.path.join(CORPUS_PATH, CATEGORY, 'lmranking.txt')
    print('LM: ', o.average_precision_at(path))
    print(10 * '-')
    path = os.path.join(CORPUS_PATH, CATEGORY, 'trmranking.txt')
    print('TRM: ', o.average_precision_at(path))
    print(10 * '-')
    path = os.path.join(CORPUS_PATH, CATEGORY, 'trlmranking.txt')
    print('TRLM: ', o.average_precision_at(path))
    print(10 * '-')