__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import os
from quora import Quora
from models.cosine import Cosine, SoftCosine

DATA_PATH='/roaming/tcastrof/quora/dataset'

class QuoraCosine(Quora):
    def __init__(self, stop=True):
        Quora.__init__(self, stop=stop)
        self.train()

    def train(self):
        self.model = Cosine()
        dict_path = os.path.join(DATA_PATH,'dict.model')
        tfidf_path = os.path.join(DATA_PATH,'tfidf.model')
        if not os.path.exists(tfidf_path):
            corpus = []
            for pair in self.trainset:
                q1 = pair['tokens1']
                corpus.append(q1)
                q2 = pair['tokens2']
                corpus.append(q2)

            self.model.init(corpus, dict_path, tfidf_path)
        else:
            self.model.load(dict_path, tfidf_path)

        del self.trainset


class QuoraSoftCosine(Quora):
    def __init__(self, stop=True, vector='word2vec'):
        Quora.__init__(self, stop=stop, vector=vector)
        self.train()

    def train(self):
        self.model = SoftCosine()
        dict_path = os.path.join(DATA_PATH,'dict.model')
        tfidf_path = os.path.join(DATA_PATH,'tfidf.model')
        if not os.path.exists(tfidf_path):
            corpus = []
            for pair in self.trainset:
                q1 = pair['tokens1']
                corpus.append(q1)
                q2 = pair['tokens2']
                corpus.append(q2)

            self.model.init(corpus, dict_path, tfidf_path)
        else:
            self.model.load(dict_path, tfidf_path)

        del self.trainset