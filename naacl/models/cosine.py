__author__='thiagocastroferreira'

import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from sklearn.metrics.pairwise import cosine_similarity

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

class Cos():
    def __init__(self):
        self.tfidf = {}
        self.dict = Dictionary()

    def init(self, traindata, path):
        self.dict = Dictionary(traindata)  # fit dictionary
        corpus = [self.dict.doc2bow(line) for line in traindata]  # convert corpus to BoW format
        self.tfidf = TfidfModel(corpus)  # fit model
        self.dict.save(os.path.join(path,'dict.model'))
        self.tfidf.save(os.path.join(path,'tfidf.model'))


    def load(self, path):
        self.dict = Dictionary.load(os.path.join(path,'dict.model'))
        self.tfidf = TfidfModel.load(os.path.join(path,'tfidf.model'))


class Cosine(Cos):
    def __init__(self):
        Cos.__init__(self)


    def dot(self, q1tfidf, q2tfidf):
        cos = 0.0
        for i, w1 in enumerate(q1tfidf):
            for j, w2 in enumerate(q2tfidf):
                if w1[0] == w2[0]:
                    cos += (w1[1] * w2[1])
        return cos


    def __call__(self, q1, q2):
        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = np.sqrt(self.dot(q1tfidf, q1tfidf))
        q2q2 = np.sqrt(self.dot(q2tfidf, q2tfidf))
        return self.dot(q1tfidf, q2tfidf) / (q1q1 * q2q2)


class SoftCosine(Cos):
    def __init__(self):
        Cos.__init__(self)

    def softdot(self, q1tfidf, q1emb, q2tfidf, q2emb):
        cos = 0.0
        for i, w1 in enumerate(q1tfidf):
            for j, w2 in enumerate(q2tfidf):
                if w1[0] == w2[0]:
                    cos += (w1[1] * w2[1])
                else:
                    m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                    cos += (w1[1] * m_ij * w2[1])
        return cos


    def __call__(self, q1, q1emb, q2, q2emb):
        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = np.sqrt(self.softdot(q1tfidf, q1emb, q1tfidf, q1emb))
        q2q2 = np.sqrt(self.softdot(q2tfidf, q2emb, q2tfidf, q2emb))
        sofcosine = self.softdot(q1tfidf, q1emb, q2tfidf, q2emb) / (q1q1 * q2q2)
        return sofcosine


    def aligndot(self, q1tfidf, q2tfidf, alignments):
        cos = 0.0
        for i, w1 in enumerate(q1tfidf):
            for j, w2 in enumerate(q2tfidf):
                if w1[0] == w2[0]:
                    cos += (w1[1] * w2[1])
                else:
                    m_ij = alignments[i][j]
                    cos += (w1[1] * m_ij * w2[1])
        return cos


    def score(self, q1, q2, t2w):
        def get_alignment(c1, c2):
            alignments = []
            for i, w in enumerate(c1):
                alignments_i = []
                w = w.lower()

                for j, t in enumerate(c2):
                    t = t.lower()
                    try:
                        w_t = t2w[t[0]][t][w[0]][w]
                    except:
                        w_t = 0.0
                    alignments_i.append(w_t)
                alignments.append(alignments_i)
            return alignments

        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = np.sqrt(self.aligndot(q1tfidf, q1tfidf, get_alignment(q1, q1)))
        q2q2 = np.sqrt(self.aligndot(q2tfidf, q2tfidf, get_alignment(q2, q2)))
        aligncosine = self.aligndot(q1tfidf, q2tfidf, get_alignment(q1, q2)) / (q1q1 * q2q2)
        return aligncosine