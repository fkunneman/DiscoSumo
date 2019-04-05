__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import json
import os
import string
punctuation = string.punctuation
import spacy
import word2vec.word2vec as w2v

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from random import shuffle

# DATA_PATH='/roaming/fkunnema/goeievraag/data/'
DATA_PATH='data/'
QUESTIONS=os.path.join(DATA_PATH, 'question_parsed_final.json')
NEW_QUESTIONS=os.path.join(DATA_PATH, 'question_parsed_proc.json')
SEEDS_PATH=os.path.join(DATA_PATH, 'seeds.json')

ANSWERS=os.path.join(DATA_PATH, 'answer_parsed.json')
NEW_ANSWERS=os.path.join(DATA_PATH, 'answer_parsed_proc.json')

CATEGORY2PARENT_PATH=os.path.join(DATA_PATH, 'qcat', 'catid2parent.json')

STOPWORDS_PATH=os.path.join(DATA_PATH, 'stopwords.txt')

TRAINDATA=os.path.join(DATA_PATH, 'ranked_questions_labeled.json')
NEW_TRAINDATA=os.path.join(DATA_PATH, 'ranked_questions_labeled_proc.json')

CORPUS_PATH=os.path.join(DATA_PATH, 'corpus.json')
DICT_PATH=os.path.join(DATA_PATH, 'dict.model')

# BM25 path
BM25_PATH=os.path.join(DATA_PATH, 'bm25.model')
# softcosine path
TFIDF_PATH=os.path.join(DATA_PATH, 'tfidf.model')
# translation path
TRANSLATION_PATH=os.path.join(DATA_PATH, 'translation.json')
# ensemble path
ENSEMBLE_PATH=os.path.join(DATA_PATH, 'ensemble.pkl')


class Initialize():
    def __init__(self, w2v_dim=300, w2v_window=10, alpha=0.7, sigma=0.3):
        self.w2v_dim = w2v_dim
        self.w2v_window = w2v_window
        self.nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])

        self.category2parent = json.load(open(CATEGORY2PARENT_PATH))

        with open(STOPWORDS_PATH) as f:
            self.stopwords = [word.lower().strip() for word in f.read().split()]

        print('Parsing questions...')
        self.init_questions()
        print('Parsing answers...')
        self.init_answers()
        print('Filtering seeds...')
        self.init_seeds()
        print('Parsing labeled data...')
        self.init_labeled_data()
        print('Parsing corpus...')
        self.init_corpus()
        print('Parsing dictionary...')
        self.init_dictionary()

        # word2vec
        print('Initializing Word2Vec...')
        self.init_word2vec()
        # translation
        print('Initializing Translation...')
        self.init_translation(alpha, sigma)
        # softcosine
        print('Initializing Softcosine...')
        self.init_sofcos()


    def init_questions(self):
        # QUESTIONS
        # if not os.path.exists(NEW_QUESTIONS):
        self.questions = {}
        questions = json.load(open(QUESTIONS))
        for i, question in enumerate(questions):
            if i % 1000 == 0:
                percentage = round(float(i+1) / len(questions), 2)
                # print('Question Progress: ', percentage, end='\r')
            text = question['questiontext']
            text = list(map(lambda token: str(token), self.nlp(text)))

            tokens_proc = [w.lower() for w in text]
            tokens_proc = [w for w in tokens_proc if w not in self.stopwords and w not in punctuation]

            self.questions[question['id']] = {
                'id': question['id'],
                'tokens_proc': tokens_proc,
                'starcount': question['starcount'],
                'answercount': question['answercount'],
                'cid': question['cid']
            }
        json.dump(self.questions, open(NEW_QUESTIONS, 'w'))
        # else:
        #     self.questions = json.load(open(NEW_QUESTIONS))


    def init_answers(self):
        self.answers = {}
        answers = json.load(open(ANSWERS))
        for i, answer in enumerate(answers):
            if i % 1000 == 0:
                percentage = round(float(i+1) / len(answers), 2)
                # print('Answer Progress: ', percentage, end='\r')
            text = answer['answertext']
            text = list(map(lambda token: str(token), self.nlp(text)))

            tokens_proc = [w.lower() for w in text]
            tokens_proc = [w for w in tokens_proc if w not in self.stopwords and w not in punctuation]

            self.answers[answer['id']] = { 'tokens_proc': tokens_proc }

        json.dump(self.answers, open(NEW_ANSWERS, 'w'))


    def init_seeds(self):
        seeds = [question for question in self.questions if int(question['answercount']) >= 1]
        seeds = [{'id': question['id'], 'tokens':question['tokens_proc'], 'category':self.category2parent[question['cid']]} for question in seeds if int(question['starcount']) >= 1]
        json.dump(seeds, open(SEEDS_PATH, 'w'))


    def init_labeled_data(self):
        procdata = json.load(open(TRAINDATA))
        self.procdata = {}
        for i, row in enumerate(procdata):
            if i % 1000 == 0:
                percentage = round(float(i+1) / len(procdata), 2)
                # print('Answer Progress: ', percentage, end='\r')
            q1id = row['id']
            q1_tokens_proc = self.questions[q1id]['tokens_proc']

            self.procdata[q1id] = {}
            for row2 in row['bm25']:
                score = float(row2['BM25-score'])
                label = 1 if row2['Lax'] == 'Similar' else 0
                q2id = row2['id']
                q2_tokens_proc = self.questions[q2id]['tokens_proc']

                self.procdata[q1id][q2id] = {
                    'q1': q1_tokens_proc,
                    'q2': q2_tokens_proc,
                    'score': score,
                    'label': label
                }
        qids = list(self.procdata.keys())
        shuffle(qids)
        trainsize = int(0.8 * len(qids))

        trainids = qids[:trainsize]
        self.traindata = {}
        for qid in trainids:
            self.traindata[qid] = self.procdata[qid]

        testids = qids[trainsize:]
        self.testdata = {}
        for qid in testids:
            self.testdata[qid] = self.procdata[qid]
        json.dump({'procdata': self.procdata, 'train': self.traindata, 'test': self.testdata}, open(NEW_TRAINDATA, 'w'))


    def init_corpus(self):
        self.corpus = []
        for qid in self.questions:
            if qid not in self.testdata:
                question = self.questions[qid]
                self.corpus.append(question['tokens_proc'])
        for answer in self.answers.values():
            self.corpus.append(answer['tokens_proc'])
        json.dump({'corpus': self.corpus}, open(CORPUS_PATH, 'w'))


    def init_dictionary(self):
        self.dict = Dictionary(self.corpus)  # fit dictionary
        self.dict.save(DICT_PATH)


    # WORD2VEC
    def init_word2vec(self):
        w2v.run(documents=self.corpus, write_path=DATA_PATH, w_dim=self.w2v_dim, window=self.w2v_window)


    # Softcosine
    def init_sofcos(self):
        corpus = [self.dict.doc2bow(line) for line in self.corpus]  # convert corpus to BoW format
        self.tfidf = TfidfModel(corpus)  # fit model
        self.tfidf.save(TFIDF_PATH)


    # Translation
    def init_translation(self, alpha, sigma):
        tokens = []
        for question in list(self.corpus):
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
        translation = { 'w_Q': w_Q, 'alpha': alpha, 'sigma': sigma }
        json.dump(translation, open(TRANSLATION_PATH, 'w'))


if __name__ == '__main__':
    Initialize()