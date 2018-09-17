__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import ev
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import re
from translation import *
from gensim import corpora
from multiprocessing import Pool

TRANSLATION_PATH='translation/model/lex.f2e'
EVALUATION_PATH='results'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

def prepare_questions(trainset):
    questions = {}

    for qid in trainset:
        question = trainset[qid]
        q1 = question['subject'] + ' ' + question['body']
        # tokenizing and removing punctuation / stopwords
        q1 = re.sub(r'[^\w\s]',' ', q1).strip()
        q1 = [w for w in nltk.word_tokenize(q1.lower()) if w not in stop]
        questions[qid] = q1

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            rel_question_id = rel_question['id']
            q2 = rel_question['subject']
            if rel_question['body']:
                q2 += ' ' + rel_question['body']
            # tokenizing and removing punctuation / stopwords
            q2 = re.sub(r'[^\w\s]',' ', q2).strip()
            q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
            questions[rel_question_id] = q2
    vocabulary = corpora.Dictionary(questions.values())
    return questions, vocabulary

def run(thread_id, questions, w_C, t2w, voclen, alpha, sigma):
    def rank(ranking):
        _ranking = []
        for i, q in enumerate(sorted(ranking, key=lambda x: x[1], reverse=True)):
            _ranking.append({'Answer_ID':q[0], 'SCORE':q[1], 'RANK':i+1, 'LABEL':'true'})
        return _ranking

    print('Load language model')
    trlm = TRLM([], w_C, t2w, voclen, alpha, sigma)  # translation-based language model

    lmranking, trmranking, trlmranking = {}, {}, {}
    for i, query_id in enumerate(questions):
        percentage = str(round((float(i+1) / len(questions)) * 100, 2)) + '%'


        question = questions[query_id]
        query = question['subject'] + ' ' + question['body']
        # tokenizing and removing punctuation / stopwords
        query = re.sub(r'[^\w\s]',' ', query).strip()
        query = [w for w in nltk.word_tokenize(query.lower()) if w not in stop]

        lmranking[query_id] = []
        trmranking[query_id] = []
        trlmranking[query_id] = []

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            rel_question_id = rel_question['id']
            rel_question_text = rel_question['subject']
            if rel_question['body']:
                rel_question_text += ' ' + rel_question['body']
            # tokenizing and removing punctuation / stopwords
            rel_question_text = re.sub(r'[^\w\s]',' ', rel_question_text).strip()
            rel_question_text = [w for w in nltk.word_tokenize(rel_question_text.lower()) if w not in stop]

            lmprob, trmprob, trlmprob, proctime = trlm.score(query, rel_question_text)
            lmranking[query_id].append((rel_question_id, lmprob))
            trmranking[query_id].append((rel_question_id, trmprob))
            trlmranking[query_id].append((rel_question_id, trlmprob))

        lmranking[query_id] = rank(lmranking[query_id])
        trmranking[query_id] = rank(trmranking[query_id])
        trlmranking[query_id] = rank(trlmranking[query_id])

        # print('Thread ID:', thread_id, 'Query Number: ', i, 'Query ID: ', query_id, 'Percentage:', percentage, sep='\t')
    return lmranking, trmranking, trlmranking

if __name__ == '__main__':
    print('Load corpus')
    trainset, devset = load.run()
    print('Preparing training questions and vocabulary')
    train_questions, vocabulary = prepare_questions(trainset)
    print('\nLoad background probabilities')
    w_C = compute_w_C(train_questions, vocabulary)  # background lm
    print('Load translation probabilities')
    t2w = translation_prob(TRANSLATION_PATH)  # translation probabilities

    if not os.path.exists(EVALUATION_PATH):
        os.mkdir(EVALUATION_PATH)

    THREADS = 25
    pool = Pool(processes=THREADS)
    processes = []

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thread_id = 0
    for alpha in alphas:
        for sigma in sigmas:
            thread_id += 1
            process = pool.apply_async(run, [thread_id, devset, w_C, t2w, len(vocabulary), alpha, sigma])
            processes.append((process, sigma, alpha))

    for row in processes:
        process, sigma, alpha = row
        lmranking, trmranking, trlmranking = process.get()

        PRED_PATH = os.path.join(EVALUATION_PATH, 'lm_alpha-' + str(alpha) + '_sigma-' + str(sigma) + '.pred')
        load.save(lmranking, PRED_PATH)
        map_model, map_baseline = ev.eval_rerankerV2(GOLD_PATH, PRED_PATH)
        print('lm_alpha-' + str(alpha) + '_sigma-' + str(sigma), 'MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), sep='\t', end='\n')

        PRED_PATH = os.path.join(EVALUATION_PATH, 'trm_alpha-' + str(alpha) + '_sigma-' + str(sigma) + '.pred')
        load.save(trmranking, PRED_PATH)
        map_model, map_baseline = ev.eval_rerankerV2(GOLD_PATH, PRED_PATH)
        print('trm_alpha-' + str(alpha) + '_sigma-' + str(sigma), 'MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), sep='\t', end='\n')

        PRED_PATH = os.path.join(EVALUATION_PATH, 'trlm_alpha-' + str(alpha) + '_sigma-' + str(sigma) + '.pred')
        load.save(trlmranking, PRED_PATH)
        map_model, map_baseline = ev.eval_rerankerV2(GOLD_PATH, PRED_PATH)
        print('trlm_alpha-' + str(alpha) + '_sigma-' + str(sigma), 'MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), sep='\t', end='\n')