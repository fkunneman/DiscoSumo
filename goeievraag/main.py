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
import numpy as np
import spacy

from gensim.summarization import bm25
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH='/roaming/fkunnema/goeievraag/parsed/'
WORD2VEC_PATH='/home/tcastrof/Question/DiscoSumo/goeievraag/word2vec/word2vec.model'
QUESTIONS='/roaming/fkunnema/goeievraag/parsed/question_parsed.json'
NEW_QUESTIONS='/roaming/fkunnema/goeievraag/parsed/question_parsed_proc.json'

ANSWERS='/roaming/fkunnema/goeievraag/parsed/answer_parsed.json'
NEW_ANSWERS='/roaming/fkunnema/goeievraag/parsed/answer_parsed_proc.json'

class GoeieVraag():
    def __init__(self):
        if not os.path.exists(NEW_QUESTIONS):
            self.parse()
        else:
            self.questions = json.load(open(NEW_QUESTIONS))
            self.answers = json.load(open(NEW_ANSWERS))

        self.nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
        self.seeds = self.filter(self.questions.values())
        # bm25
        self.init_bm25([question[1] for question in self.seeds])
        # word2vec
        self.init_word2vec()
        # softcosine
        self.init_sofcos()


    def __call__(self, query):
        query_proc = list(map(lambda token: str(token).lower(), self.nlp(query)))
        query_proc = [w for w in query_proc if w not in stopwords and w not in punctuation]

        # retrieve 30 candidates with bm25
        questions = self.retrieve(query_proc)
        # reranking with softcosine
        questions = self.rerank(query_proc, questions)

        result = { 'query': query, 'questions': [] }
        for question in questions:
            question_id, _, score = question
            q = self.questions[question_id]
            q['score'] = score
            result['questions'].append(q)

        bestanswer_id = self.questions[questions[0][0]]['bestanswer']
        result['bestanswer'] = self.answers[bestanswer_id]
        return result


    def parse(self):
        self.questions = json.load(open(QUESTIONS))
        for i, question in enumerate(self.questions):
            if i % 1000 == 0:
                percentage = round(float(i+1) / len(self.questions), 2)
                print('Question Progress: ', percentage)
            text = question['questiontext']
            text = list(map(lambda token: str(token).lower(), self.nlp(text)))

            question['tokens'] = text
            question['tokens_proc'] = [w for w in text if w not in stopwords and w not in punctuation]
        self.questions = dict([(question['id'], question) for question in self.questions])

        self.answers = json.load(open(ANSWERS))
        for i, answer in enumerate(self.answers):
            if i % 1000 == 0:
                percentage = round(float(i+1) / len(self.answers), 2)
                print('Answer Progress: ', percentage)
            text = answer['answertext']
            text = list(map(lambda token: str(token).lower(), self.nlp(text)))

            answer['tokens'] = text
            answer['tokens_proc'] = [w for w in text if w not in stopwords and w not in punctuation]
        self.answers = dict([(answer['id'], answer) for answer in self.answers])

        json.dump(self.questions, open(NEW_QUESTIONS, 'w'))
        json.dump(self.answers, open(NEW_ANSWERS, 'w'))
        return self.questions, self.answers


    def filter(self, questions):
        starcount = [float(question['starcount']) for question in questions]
        avgstar = sum(starcount) / len(starcount)

        answercounts = [float(question['answercount']) for question in questions]
        avganswer = sum(answercounts) / len(answercounts)

        seeds = [question for question in questions if int(question['answercount']) >= int(avganswer)]
        seeds = [(question['id'], question['tokens_proc']) for question in seeds if int(question['starcount']) > avgstar]
        return seeds


    # BM25
    def init_bm25(self, corpus):
        self.model = bm25.BM25(corpus)

        # get average idf
        self.average_idf = sum(map(lambda k: float(self.model.idf[k]), self.model.idf.keys())) / len(self.model.idf.keys())


    def retrieve(self, query, n=30):
        scores = self.model.get_scores(query, self.average_idf)
        questions = [(self.seeds[i][0], self.seeds[i][1], scores[i]) for i in range(len(self.seeds))]
        questions = sorted(questions, key=lambda x: x[2], reverse=True)[:n]
        return questions


    # WORD2VEC
    def init_word2vec(self):
        self.word2vec = Word2Vec.load(WORD2VEC_PATH)


    def encode(self, question):
        emb = []
        for w in question:
            try:
                emb.append(self.word2vec[w.lower()])
            except:
                emb.append(300 * [0])
        return emb


    # Softcosine
    def init_sofcos(self):
        if not os.path.exists(os.path.join(DATA_PATH,'tfidf.model')):
            corpus = []
            for question in self.questions.values():
                corpus.append(question['tokens'])
            for answer in self.answers.values():
                corpus.append(answer['tokens'])

            self.dict = Dictionary(corpus)  # fit dictionary
            corpus = [self.dict.doc2bow(line) for line in corpus]  # convert corpus to BoW format
            self.tfidf = TfidfModel(corpus)  # fit model
            self.dict.save(os.path.join(DATA_PATH, 'dict.model'))
            self.tfidf.save(os.path.join(DATA_PATH, 'tfidf.model'))
        else:
            self.dict = Dictionary.load(os.path.join(DATA_PATH, 'dict.model'))
            self.tfidf = TfidfModel.load(os.path.join(DATA_PATH, 'tfidf.model'))


    def softcos(self, q1, q2):
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

        if type(q1) == str:
            q1 = q1.split()
        if type(q2) == str:
            q2 = q2.split()

        q1emb = self.encode(q1)
        q2emb = self.encode(q2)

        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = np.sqrt(dot(q1tfidf, q1emb, q1tfidf, q1emb))
        q2q2 = np.sqrt(dot(q2tfidf, q2emb, q2tfidf, q2emb))
        sofcosine = dot(q1tfidf, q1emb, q2tfidf, q2emb) / (q1q1 * q2q2)
        return sofcosine


    def rerank(self, query, questions, n=10):
        for question in questions:
            question[2] = self.softcos(query, question[1])

        questions = sorted(questions, key=lambda x: x[2], reverse=True)[:n]
        return questions

if __name__ == '__main__':
    pass
