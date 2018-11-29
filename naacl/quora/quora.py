__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import elmo.elmo as elmo
import json
import numpy as np
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import preprocessing
import word2vec.word2vec as word2vec

ALIGNMENTS_PATH='/roaming/tcastrof/quora/alignments/model/lex.f2e'
WORD2VEC_PATH='/roaming/tcastrof/quora/word2vec/word2vec.model'
ELMO_PATH='/roaming/tcastrof/quora/elmo/'

DATA_PATH='/roaming/tcastrof/quora/dataset'
TRAIN_PATH=os.path.join(DATA_PATH, 'train.data')
DEV_PATH=os.path.join(DATA_PATH, 'dev.data')
TEST_PATH=os.path.join(DATA_PATH, 'test.data')

class Quora():
    def __init__(self, stop=True, vector=''):
        if not os.path.exists(DEV_PATH):
            preprocessing.run()

        self.stop = stop
        self.vector = vector

        logging.info('Preparing test set...')
        self.testset = json.load(open(TEST_PATH))

        logging.info('Preparing development set...')
        self.devset = json.load(open(DEV_PATH))

        logging.info('Preparing trainset...')
        self.trainset = json.load(open(TRAIN_PATH))

        self.word2vec = None
        if 'word2vec' in self.vector:
            self.word2vec = word2vec.init_word2vec(WORD2VEC_PATH)

        self.trainidx = self.trainelmo = self.devidx = self.develmo = self.testidx = self.testelmo = None
        self.fulltrainidx = self.fulltrainelmo = self.fulldevidx = self.fulldevelmo = self.fulltestidx = self.fulltestelmo = None
        if 'elmo' in self.vector:
            self.trainidx, self.trainelmo, self.devidx, self.develmo, self.testidx, self.testelmo = elmo.init_elmo(True, ELMO_PATH)
            # self.fulltrainidx, self.fulltrainelmo, self.fulldevidx, self.fulldevelmo, self.fulltestidx, self.fulltestelmo = elmo.init_elmo(False, ELMO_PATH)

        # self.alignments = self.init_alignments(ALIGNMENTS_PATH)


    def init_alignments(self, path):
        with open(path) as f:
            doc = list(map(lambda x: x.split(), f.read().split('\n')))

        alignments = {}
        for row in doc[:-1]:
            t = row[0]
            if t[0] not in alignments:
                alignments[t[0]] = {}
            if t not in alignments[t[0]]:
                alignments[t[0]][t] = {}

            w = row[1]
            if w[0] not in alignments[t[0]][t]:
                alignments[t[0]][t][w[0]] = {}

            prob = float(row[2])
            alignments[t[0]][t][w[0]][w] = prob
        return alignments


    def encode(self, qid, q, elmoidx, elmovec):
        def w2v():
            emb = []
            for w in q:
                try:
                    emb.append(self.word2vec[w.lower()])
                except:
                    emb.append(300 * [0])
            return emb

        def elmo():
            return elmovec.get(str(elmoidx[qid]))

        if self.vector == 'word2vec':
            return w2v()
        elif self.vector == 'elmo':
            return elmo()
        elif self.vector == 'word2vec+elmo':
            w2vemb = w2v()
            elmoemb = elmo()
            return [np.concatenate([w2vemb[i], elmoemb[i]]) for i in range(len(w2vemb))]
        return 0


    def load(self, path):
        with open(path) as f:
            doc = f.read().split('\n')
        parameter_settings, doc = doc[0], doc[1:]
        ranking = {}
        for row in doc:
            q1id, q2id, _, score, label = row.strip().split('\t')
            score = float(score)
            label = 1 if label == 'true' else 0
            if q1id not in ranking: ranking[q1id] = []

            ranking[q1id].append((label, score, q2id))
        return ranking


    def save(self, ranking, path, parameter_settings):
        with open(path, 'w') as f:
            f.write(parameter_settings)
            f.write('\n')
            for q1id in ranking:
                for row in ranking[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))