__author__='thiagocastroferreira'

import features
import json
import load
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import numpy as np
import os
import utils

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from sklearn.metrics.pairwise import cosine_similarity

from stanfordcorenlp import StanfordCoreNLP

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': 'localhost', 'user': 'tcastrof'}
logger = logging.getLogger('tcpserver')

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

DATA_PATH='data'

TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class SemevalCosine():
    def __init__(self, trainset, devset, testset):
        props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
        corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

        logging.info('Preparing development set...', extra=d)
        if not os.path.exists(DEV_PATH):
            self.devset = utils.prepare_corpus(devset, corenlp=corenlp, props=props)
            json.dump(self.devset, open(DEV_PATH, 'w'))
        else:
            self.devset = json.load(open(DEV_PATH))

        logging.info('Preparing trainset...', extra=d)
        if not os.path.exists(TRAIN_PATH):
            self.trainset = utils.prepare_corpus(trainset, corenlp=corenlp, props=props)
            json.dump(self.trainset, open(TRAIN_PATH, 'w'))
        else:
            self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info)
        logging.info('Preparing test set...', extra=d)
        self.testset = utils.prepare_corpus(testset, corenlp=corenlp, props=props)

        corenlp.close()

        logging.info('Preparing word2vec...', extra=d)
        self.word2vec = features.init_word2vec()
        # self.glove, self.voc2id, self.id2voc = features.init_glove()
        logging.info('Preparing elmo...', extra=d)
        self.trainidx, self.trainelmo, self.devidx, self.develmo = features.init_elmo()


    def train(self):
        traindata = []
        for qid in self.trainset:
            query = self.trainset[qid]['tokens_full']

            duplicates = self.trainset[qid]['duplicates']
            for duplicate in duplicates:
                question = duplicate['rel_question']['tokens_full']
                traindata.append(question)

            traindata.append(query)

        self.dict = Dictionary(traindata)  # fit dictionary
        corpus = [self.dict.doc2bow(line) for line in traindata]  # convert corpus to BoW format
        self.tfidf = TfidfModel(corpus)  # fit model


    def dot(self, q1, q2):
        cos = 0.0
        for i, w1 in enumerate(q1):
            for j, w2 in enumerate(q2):
                if w1[0] == w2[0]:
                    if q1[i] not in stop and q2[j] not in stop:
                        cos += (w1[1] * w2[1])
        return cos


    def simple_score(self, q1, q2):
        if type(q1) == str:
            q1 = q1.split()
        if type(q2) == str:
            q2 = q2.split()

        q1 = self.tfidf[self.dict.doc2bow(q1)]
        q2 = self.tfidf[self.dict.doc2bow(q2)]
        q1q1 = np.sqrt(self.dot(q1, q1))
        q2q2 = np.sqrt(self.dot(q2, q2))
        return self.dot(q1, q2) / (q1q1 * q2q2)


    def softdot(self, q1, q1emb, q2, q2emb):
        cos = 0.0
        for i, w1 in enumerate(q1):
            for j, w2 in enumerate(q2):
                if q1[i] not in stop and q2[j] not in stop:
                    m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                    # m_ij = cosine_similarity([q1emb[i]], [q2emb[j]])[0][0]
                    cos += (w1[1] * m_ij * w2[1])
        return cos


    def score(self, q1, q1emb, q2, q2emb):
        if type(q1) == str:
            q1 = q1.split()
        if type(q2) == str:
            q2 = q2.split()

        q1 = self.tfidf[self.dict.doc2bow(q1)]
        q2 = self.tfidf[self.dict.doc2bow(q2)]
        q1q1 = np.sqrt(self.softdot(q1, q1emb, q1, q1emb))
        q2q2 = np.sqrt(self.softdot(q2, q2emb, q2, q2emb))
        sofcosine = self.softdot(q1, q1emb, q2, q2emb) / (q1q1 * q2q2)
        return sofcosine


    def validate(self):
        logging.info('Validating', extra=d)
        softranking, simranking = {}, {}
        for j, q1id in enumerate(self.devset):
            softranking[q1id], simranking[q1id] = [], []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_full']
            elmo_emb1 = self.develmo.get(str(self.devidx[q1id]))
            w2v_emb = features.encode(q1, self.word2vec)
            # q1emb = features.glove_encode(q1, self.glove, self.voc2id)
            q1emb = [np.concatenate([w2v_emb[i], elmo_emb1[i]]) for i in range(len(w2v_emb))]

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_full']
                elmo_emb2 = self.develmo.get(str(self.devidx[q2id]))
                w2v_emb = features.encode(q2, self.word2vec)
                # q2emb = features.glove_encode(q2, self.glove, self.voc2id)
                q2emb = [np.concatenate([w2v_emb[i], elmo_emb2[i]]) for i in range(len(w2v_emb))]

                simple_score = self.simple_score(q1, q2)
                score = self.score(q1, q1emb, q2, q2emb)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                simranking[q1id].append((real_label, simple_score, q2id))
                softranking[q1id].append((real_label, score, q2id))

        with open('data/softranking.txt', 'w') as f:
            for q1id in softranking:
                for row in softranking[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))

        logging.info('Finishing to validate.', extra=d)
        return softranking, simranking

if __name__ == '__main__':
    logging.info('Load corpus', extra=d)
    trainset, devset = load.run()

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # Softcosine
    semeval = SemevalCosine(trainset, devset, [])
    semeval.train()

    softranking, simranking = semeval.validate()
    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, softranking)
    print('Evaluation Soft-Cosine')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, simranking)
    print('Evaluation Cosine')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')