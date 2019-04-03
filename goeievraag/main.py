__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import json
import os
import string
punctuation = string.punctuation
import pandas as pd
import numpy as np
import spacy
import word2vec.word2vec as w2v

from category.qcat import QCat

from classifier import Model

from gensim.summarization import bm25
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from random import shuffle

from sklearn.metrics.pairwise import cosine_similarity

# DATA_PATH='/roaming/fkunnema/goeievraag/data/'
DATA_PATH='data/'
WORD2VEC_PATH='word2vec/'
QUESTIONS=os.path.join(DATA_PATH, 'question_parsed.json')
NEW_QUESTIONS=os.path.join(DATA_PATH, 'question_parsed_proc.json')

ANSWERS=os.path.join(DATA_PATH, 'answer_parsed.json')
NEW_ANSWERS=os.path.join(DATA_PATH, 'answer_parsed_proc.json')

STOPWORDS_PATH=os.path.join(DATA_PATH, 'stopwords.txt')

TRAINDATA=os.path.join(DATA_PATH, 'ranked_questions_labeled.json')
NEW_TRAINDATA=os.path.join(DATA_PATH, 'ranked_questions_labeled_proc.json')

CORPUS_PATH=os.path.join(DATA_PATH, 'corpus.json')
DICT_PATH=os.path.join(DATA_PATH, 'dict.model')

CATEGORY2PARENT_PATH=os.path.join(DATA_PATH, 'qcat', 'catid2parent.json')
CATEGORY_MODEL_PATH=os.path.join(DATA_PATH, 'qcat', 'questions.model.pkl')
LABEL_ENCODER_PATH=os.path.join(DATA_PATH, 'qcat', 'questions.le')
CATEGORY2ID_PATH=os.path.join(DATA_PATH, 'qcat', 'cat2id_dict.txt')
VOCABULARY_PATH=os.path.join(DATA_PATH, 'qcat', 'questions.featureselection.txt')

# softcosine path
TFIDF_PATH=os.path.join(DATA_PATH, 'tfidf.model')
# translation path
TRANSLATION_PATH=os.path.join(DATA_PATH, 'translation.json')
# ensemble path
ENSEMBLE_PATH=os.path.join(DATA_PATH, 'ensemble.pkl')


class GoeieVraag():
    def __init__(self, w2v_dim=300, w2v_window=10, alpha=0.7, sigma=0.3, evaluation=False):
        self.w2v_dim = w2v_dim
        self.w2v_window = w2v_window
        self.nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
        print('Parsing questions and answers...')
        self.init_datasets()

        # category model
        print('Initializing Categorization model...')
        self.category2parent = json.load(open(CATEGORY2PARENT_PATH))
        self.categories = list(set(self.category2parent.values()))
        self.question2cat = QCat(CATEGORY_MODEL_PATH, LABEL_ENCODER_PATH, CATEGORY2ID_PATH, VOCABULARY_PATH)

        print('Filter seed questions...')
        if evaluation:
            self.seeds = [{'id': question['id'], 'tokens':question['tokens_proc'], 'category':self.category2parent[question['cid']]} for question in self.questions.values()]
        else:
            self.seeds = self.init_seeds(self.questions.values())

        # bm25
        print('Initializing BM25...')
        self.init_bm25(self.seeds)
        # word2vec
        print('Initializing Word2Vec...')
        self.init_word2vec()
        # translation
        print('Initializing Translation...')
        self.init_translation(alpha, sigma)
        # softcosine
        print('Initializing Softcosine...')
        self.init_sofcos()
        # ensemble
        print('Initializing Ensemble...')
        self.init_ensemble()


    def __call__(self, query, method='ensemble'):
        # tokenize and lowercase
        tokens = list(map(lambda token: str(token).lower(), self.nlp(query)))
        # remove stop words and punctuation
        query_proc = [w for w in tokens if w not in self.stopwords and w not in punctuation]

        # retrieve the 5 most likely categories of the query
        categories = [c[1] for c in self.question2cat(' '.join(tokens))]
        # retrieve 30 candidates with bm25
        questions = self.retrieve(query_proc, categories)
        # reranking with chosen method
        if method != 'bm25':
            questions = self.rerank(query=query_proc, questions=questions, method=method)

        result = { 'query': query, 'questions': [] }
        for question in questions:
            question_id = question['id']
            q = self.questions[question_id]
            q['score'] = question['score'] if method == 'bm25' else question['rescore']
            result['questions'].append(q)

        # bestanswer_id = self.questions[questions[0][0]]['bestanswer']
        # result['bestanswer'] = self.answers[bestanswer_id]
        return result


    def init_datasets(self):
        # STOPWORDS
        with open(STOPWORDS_PATH) as f:
            self.stopwords = [word.lower().strip() for word in f.read().split()]

        # QUESTIONS
        if not os.path.exists(NEW_QUESTIONS):
            self.questions = json.load(open(QUESTIONS))
            for i, question in enumerate(self.questions):
                if i % 1000 == 0:
                    percentage = round(float(i+1) / len(self.questions), 2)
                    # print('Question Progress: ', percentage, end='\r')
                text = question['questiontext']
                text = list(map(lambda token: str(token), self.nlp(text)))

                question['tokens'] = text
                question['tokens_proc'] = [w.lower() for w in text]
                question['tokens_proc'] = [w for w in question['tokens_proc'] if w not in self.stopwords and w not in punctuation]
            self.questions = dict([(question['id'], question) for question in self.questions])
            json.dump(self.questions, open(NEW_QUESTIONS, 'w'))
        else:
            self.questions = json.load(open(NEW_QUESTIONS))

        # ANSWERS
        if not os.path.exists(NEW_ANSWERS):
            self.answers = json.load(open(ANSWERS))
            for i, answer in enumerate(self.answers):
                if i % 1000 == 0:
                    percentage = round(float(i+1) / len(self.answers), 2)
                    # print('Answer Progress: ', percentage, end='\r')
                text = answer['answertext']
                text = list(map(lambda token: str(token), self.nlp(text)))

                answer['tokens'] = text
                answer['tokens_proc'] = [w.lower() for w in text]
                answer['tokens_proc'] = [w for w in answer['tokens_proc'] if w not in self.stopwords and w not in punctuation]
            self.answers = dict([(answer['id'], answer) for answer in self.answers])
            json.dump(self.answers, open(NEW_ANSWERS, 'w'))
        else:
            self.answers = json.load(open(NEW_ANSWERS))

        # TRAINDATA
        if not os.path.exists(NEW_TRAINDATA):
            procdata = json.load(open(TRAINDATA))
            self.procdata = {}
            for i, row in enumerate(procdata):
                if i % 1000 == 0:
                    percentage = round(float(i+1) / len(procdata), 2)
                    # print('Answer Progress: ', percentage, end='\r')
                q1id = row['id']
                q1_tokens = self.questions[q1id]['tokens']
                q1_tokens_proc = self.questions[q1id]['tokens_proc']

                self.procdata[q1id] = {}
                for row2 in row['bm25']:
                    score = float(row2['BM25-score'])
                    label = 1 if row2['Lax'] == 'Similar' else 0
                    q2id = row2['id']

                    q2_tokens = self.questions[q2id]['tokens']
                    q2_tokens_proc = self.questions[q2id]['tokens_proc']

                    self.procdata[q1id][q2id] = {
                        'q1': q1_tokens_proc,
                        'q1_full': q1_tokens,
                        'q2': q2_tokens_proc,
                        'q2_full': q2_tokens,
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
            del self.procdata
            json.dump({'train': self.traindata, 'test': self.testdata}, open(NEW_TRAINDATA, 'w'))
        else:
            procdata = json.load(open(NEW_TRAINDATA))
            self.traindata, self.testdata = procdata['train'], procdata['test']

        # CORPUS = QUESTIONS + ANSWERS
        if not os.path.exists(CORPUS_PATH):
            self.corpus = []
            for qid in self.questions:
                if qid not in self.testdata:
                    question = self.questions[qid]
                    self.corpus.append(question['tokens_proc'])
            for answer in self.answers.values():
                self.corpus.append(answer['tokens_proc'])
            json.dump({'corpus': self.corpus}, open(CORPUS_PATH, 'w'))
        else:
            self.corpus = json.load(open(CORPUS_PATH))['corpus']

        # DICTIONARY
        if not os.path.exists(DICT_PATH):
            self.dict = Dictionary(self.corpus)  # fit dictionary
            self.dict.save(DICT_PATH)
        else:
            self.dict = Dictionary.load(DICT_PATH)


    def init_seeds(self, questions):
        starcount = [float(question['starcount']) for question in questions]
        avgstar = sum(starcount) / len(starcount)

        answercounts = [float(question['answercount']) for question in questions]
        avganswer = sum(answercounts) / len(answercounts)

        seeds = [question for question in questions if int(question['answercount']) >= int(avganswer)]
        seeds = [{'id': question['id'], 'tokens':question['tokens_proc'], 'category':self.category2parent[question['cid']]} for question in seeds if int(question['starcount']) > avgstar]
        return seeds


    # BM25
    def init_bm25(self, corpus):
        ids, questions = [], []
        for row in corpus:
            ids.append(row['id'])
            questions.append(row['tokens'])


        self.idx2id = dict([(i, qid) for i, qid in enumerate(ids)])
        self.id2idx = dict([(qid, i) for i, qid in enumerate(ids)])
        self.bm25 = bm25.BM25(questions)


    def retrieve(self, query, categories, n=30):
        result = []
        scores = self.bm25.get_scores(query)
        questions = []
        for i in range(len(self.seeds)):
            questions.append({
                'id': self.seeds[i]['id'],
                'tokens': self.seeds[i]['tokens'],
                'category': self.seeds[i]['category'],
                'score': scores[i]
            })

        # retrieve the n / 5 most likely questions for each category
        n_ = int(n / 5)
        for cid in categories:
            fquestions = [question for question in questions if question['category'] == cid]
            fquestions = sorted(fquestions, key=lambda x: x['score'], reverse=True)[:n_]
            result.extend(fquestions)
        return result


    # WORD2VEC
    def init_word2vec(self):
        fname = 'word2vec.' + str(self.w2v_dim) + '_' + str(self.w2v_window) + '.model'
        path = os.path.join(WORD2VEC_PATH, fname)
        if not os.path.exists(path):
            w2v.run(question_path=NEW_QUESTIONS, answer_path=NEW_ANSWERS, write_path=WORD2VEC_PATH, w_dim=self.w2v_dim, window=self.w2v_window)
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
    def init_sofcos(self):
        if not os.path.exists(TFIDF_PATH):
            corpus = [self.dict.doc2bow(line) for line in self.corpus]  # convert corpus to BoW format
            self.tfidf = TfidfModel(corpus)  # fit model
            self.tfidf.save(TFIDF_PATH)
        else:
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
    def init_translation(self, alpha, sigma):
        if not os.path.exists(TRANSLATION_PATH):
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
        else:
            translation = json.load(open(TRANSLATION_PATH))
            w_Q = translation['w_Q']
            alpha = translation['alpha']
            sigma = translation['sigma']
        self.prob_w_C = w_Q
        self.alpha = alpha
        self.sigma = sigma


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

        if not os.path.exists(ENSEMBLE_PATH):
            ids, questions = [], []
            for row in self.questions.values():
                ids.append(row['id'])
                questions.append(row['tokens'])


            id2idx = dict([(qid, i) for i, qid in enumerate(ids)])
            bm25_ = bm25.BM25(questions)

            traindata = self.traindata

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
            self.ensemble.save(ENSEMBLE_PATH)
        else:
            self.ensemble.load(ENSEMBLE_PATH)


    def ensembling(self, q1, q1emb, q2id, q2, q2emb):
        bm25score = self.bm25.get_score(q1, self.id2idx[q2id])
        translation = self.translate(q1, q1emb, q2, q2emb)
        softcosine = self.softcos(q1, q1emb, q2, q2emb)

        X = [bm25score, translation, softcosine]
        X = self.ensemble.scale([X])[0]
        clfscore, pred_label = self.ensemble.score(X)
        return clfscore, pred_label


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


if __name__ == '__main__':
    model = GoeieVraag()
    result = model('wat is kaas?')
    print(result)
