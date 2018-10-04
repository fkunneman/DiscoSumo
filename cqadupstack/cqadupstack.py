__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/cqadupstack/CQADupStack')
sys.path.append('../')
import query_cqadupstack as qcqa

import os
import re
from translation import *
from gensim import corpora
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
    t2w = translation_prob(TRANSLATION_PATH)  # translation probabilities

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
