__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import json
import load
import os
stopwords = load.load_stopwords()
import string
punctuation = string.punctuation
import pandas as pd
import numpy as np
import spacy

from category.qcat import QCat

from classifier import Model

from gensim.summarization import bm25
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from random import shuffle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# DATA_PATH='/roaming/fkunnema/goeievraag/parsed/'
# WORD2VEC_PATH='/home/tcastrof/Question/DiscoSumo/goeievraag/word2vec'
# QUESTIONS='/roaming/fkunnema/goeievraag/parsed/question_parsed.json'
# NEW_QUESTIONS='/roaming/fkunnema/goeievraag/parsed/question_parsed_proc.json'
#
# ANSWERS='/roaming/fkunnema/goeievraag/parsed/answer_parsed.json'
# NEW_ANSWERS='/roaming/fkunnema/goeievraag/parsed/answer_parsed_proc.json'
#
# TRAINDATA='/roaming/fkunnema/goeievraag/exp_similarity/ranked_questions_labeled.json'
# NEW_TRAINDATA='/roaming/fkunnema/goeievraag/exp_similarity/ranked_questions_labeled_proc.json'
#
# DICT_PATH='/roaming/fkunnema/goeievraag/parsed/dict.model'
#
# CATEGORY2PARENT_PATH='/roaming/fkunnema/goeievraag/qcat/catid2parent.json'
# CATEGORY_MODEL_PATH='/roaming/fkunnema/goeievraag/qcat/questions.model.pkl'
# LABEL_ENCODER_PATH='/roaming/fkunnema/goeievraag/qcat/questions.le'
# CATEGORY2ID_PATH='/roaming/fkunnema/goeievraag/qcat/cat2id_dict.txt'
# VOCABULARY_PATH='/roaming/fkunnema/goeievraag/qcat/questions.featureselection.txt'

DATA_PATH='data/'
WORD2VEC_PATH='word2vec/'
QUESTIONS='data/question_parsed.json'
NEW_QUESTIONS='data/question_parsed_proc.json'

ANSWERS='data/answer_parsed.json'
NEW_ANSWERS='data/answer_parsed_proc.json'

TRAINDATA='data/ranked_questions_labeled.json'
NEW_TRAINDATA='data/ranked_questions_labeled_proc.json'

DICT_PATH='data/dict.model'

CATEGORY2PARENT_PATH='data/qcat/catid2parent.json'
CATEGORY_MODEL_PATH='data/qcat/questions.model.pkl'
LABEL_ENCODER_PATH='data/qcat/questions.le'
CATEGORY2ID_PATH='data/qcat/cat2id_dict.txt'
VOCABULARY_PATH='data/qcat/questions.featureselection.txt'

class GoeieVraag():
    def __init__(self, w2v_dim=300, alpha=0.7, sigma=0.3, evaluation=False):
        self.w2v_dim = w2v_dim
        self.nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
        print('Parsing questions and answers...')
        self.parse()

        # category model
        print('Initializing Categorization model...')
        self.category2parent = json.load(open(CATEGORY2PARENT_PATH))
        self.categories = list(set(self.category2parent.values()))
        self.question2cat = QCat(CATEGORY_MODEL_PATH, LABEL_ENCODER_PATH, CATEGORY2ID_PATH, VOCABULARY_PATH)

        print('Filter seed questions...')
        if evaluation:
            self.seeds_ = [{'id': question['id'], 'tokens':question['tokens_proc'], 'category':self.category2parent[question['cid']]} for question in self.questions.values()]
        else:
            self.seeds_ = self.filter(self.questions.values())
        self.seeds = dict([(cid, [seed for seed in self.seeds_ if seed['category'] == cid]) for cid in self.categories])

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


    def __call__(self, query):
        tokens = list(map(lambda token: str(token).lower(), self.nlp(query)))
        query_proc = [w for w in tokens if w not in stopwords and w not in punctuation]

        # retrieve the 5 most likely categories of the query
        categories = [c[1] for c in self.question2cat(' '.join(tokens))]
        # retrieve 30 candidates with bm25
        questions = self.retrieve(query_proc, categories)
        # reranking with softcosine
        questions = self.rerank(query_proc, questions)

        result = { 'query': query, 'questions': [] }
        for question in questions:
            question_id, score = question['id'], question['rescore']
            q = self.questions[question_id]
            q['score'] = score
            result['questions'].append(q)

        # bestanswer_id = self.questions[questions[0][0]]['bestanswer']
        # result['bestanswer'] = self.answers[bestanswer_id]
        return result


    def parse(self):
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
                question['tokens_proc'] = [w for w in question['tokens_proc'] if w not in stopwords and w not in punctuation]
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
                answer['tokens_proc'] = [w for w in answer['tokens_proc'] if w not in stopwords and w not in punctuation]
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

        self.corpus = []
        for qid in self.questions:
            if qid not in self.testdata:
                question = self.questions[qid]
                self.corpus.append(question['tokens_proc'])
        for answer in self.answers.values():
            self.corpus.append(answer['tokens_proc'])
        if not os.path.exists(DICT_PATH):
            self.dict = Dictionary(self.corpus)  # fit dictionary
            self.dict.save(DICT_PATH)
        else:
            self.dict = Dictionary.load(DICT_PATH)


    def filter(self, questions):
        starcount = [float(question['starcount']) for question in questions]
        avgstar = sum(starcount) / len(starcount)

        answercounts = [float(question['answercount']) for question in questions]
        avganswer = sum(answercounts) / len(answercounts)

        seeds = [question for question in questions if int(question['answercount']) >= int(avganswer)]
        seeds = [{'id': question['id'], 'tokens':question['tokens_proc'], 'category':self.category2parent[question['cid']]} for question in seeds if int(question['starcount']) > avgstar]
        return seeds


    # BM25
    def init_bm25(self, corpus):
        self.id2idx = dict([(cid, {}) for cid in self.categories])
        self.idx2id = dict([(cid, {}) for cid in self.categories])
        self.bm25 = dict([(cid, None) for cid in self.categories])
        self.average_idf = dict([(cid, 0.0) for cid in self.categories])

        for cid in corpus:
            ids, questions = [], []
            for row in corpus[cid]:
                ids.append(row['id'])
                questions.append(row['tokens'])

            self.idx2id[cid] = dict([(i, qid) for i, qid in enumerate(ids)])
            self.id2idx[cid] = dict([(qid, i) for i, qid in enumerate(ids)])
            self.bm25[cid] = bm25.BM25(questions)

            # get average idf
            self.average_idf[cid] = sum(map(lambda k: float(self.bm25[cid].idf[k]), self.bm25[cid].idf.keys())) / len(self.bm25[cid].idf.keys())

        ids, questions = [], []
        for row in self.seeds_:
            ids.append(row['id'])
            questions.append(row['tokens'])

        self.idx2id_ = dict([(i, qid) for i, qid in enumerate(ids)])
        self.id2idx_ = dict([(qid, i) for i, qid in enumerate(ids)])
        self.bm25_ = bm25.BM25(questions)

        # get average idf
        self.average_idf_ = sum(map(lambda k: float(self.bm25_.idf[k]), self.bm25_.idf.keys())) / len(self.bm25_.idf.keys())


    def retrieve(self, query, categories, n=30):
        result = []
        for cid in categories:
            scores = self.bm25[cid].get_scores(query, self.average_idf[cid])
            questions = []
            for i in range(len(self.seeds[cid])):
                questions.append({
                    'id': self.seeds[cid][i]['id'],
                    'tokens': self.seeds[cid][i]['tokens'],
                    'score': scores[i]
                })
            # questions = [(self.seeds[cid][i][0], self.seeds[cid][i][1], self.idx2id[cid][i], scores[i]) for i in range(len(self.seeds[cid]))]
            n_ = int(n / 5)
            questions = sorted(questions, key=lambda x: x['score'], reverse=True)[:n_]
            result.extend(questions)
        return result


    # WORD2VEC
    def init_word2vec(self):
        fname = 'word2vec.' + str(self.w2v_dim) + '.model'
        path = os.path.join(WORD2VEC_PATH, fname)
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
        if not os.path.exists(os.path.join(DATA_PATH,'tfidf.model')):
            corpus = [self.dict.doc2bow(line) for line in self.corpus]  # convert corpus to BoW format
            self.tfidf = TfidfModel(corpus)  # fit model
            self.tfidf.save(os.path.join(DATA_PATH, 'tfidf.model'))
        else:
            self.tfidf = TfidfModel.load(os.path.join(DATA_PATH, 'tfidf.model'))


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
        traindata = self.traindata
        self.ensemble = Model()

        X, y = [], []
        for q1id in traindata:
            for q2id in traindata[q1id]:
                pair = traindata[q1id][q2id]
                q1 = pair['q1']
                q2 = pair['q2']
                label = pair['label']

                q1, q1emb, q2, q2emb = self.preprocess(q1, q2)

                # TCF: ATTENTION ON THE ID HERE. WE NEED TO CHECK THIS
                bm25 = self.bm25_.get_score(q1, self.id2idx_[q2id], self.average_idf_)
                translation = self.translate(q1, q1emb, q2, q2emb)
                softcosine = self.softcos(q1, q1emb, q2, q2emb)

                X.append([bm25, translation, softcosine])
                y.append(label)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.ensemble.train_regression(trainvectors=X, labels=y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def ensembling(self, q1, q1emb, q2id, q2, q2emb):
        bm25 = self.bm25_.get_score(q1, self.id2idx_[q2id], self.average_idf_)
        translation = self.translate(q1, q1emb, q2, q2emb)
        softcosine = self.softcos(q1, q1emb, q2, q2emb)

        X = [bm25, translation, softcosine]
        X = self.scaler.transform([X])[0]
        clfscore, pred_label = self.ensemble.score(X)
        return clfscore, pred_label


    def rerank(self, query, questions, n=10):
        for i, question in enumerate(questions):
            q1, q1emb, q2, q2emb = self.preprocess(query, question['tokens'])

            questions[i]['rescore'], _ = self.ensembling(q1, q1emb, question['id'], q2, q2emb)

        questions = sorted(questions, key=lambda x: x['rescore'], reverse=True)[:n]
        return questions


if __name__ == '__main__':
    model = GoeieVraag()
    result = model('wat is kaas?')
    print(result)
