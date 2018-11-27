__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import os
from quora import Quora
from gensim.corpora import Dictionary
from models.translation import TRLM, compute_w_C

DATA_PATH='/roaming/tcastrof/quora/dataset'

class QuoraTranslations(Quora):
    def __init__(self, alpha, sigma, stop=True, vector='word2vec'):
        Quora.__init__(self, stop=stop, vector=vector)
        self.alpha = alpha
        self.sigma = sigma
        self.train()

    def train(self):
        questions = []
        for i, pair in enumerate(self.trainset):
            q1 = pair['tokens1']
            questions.append(q1)

            q2 = pair['tokens2']
            questions.append(q2)

        path = os.path.join(DATA_PATH, 'transdict.model')
        if not os.path.exists(path):
            self.vocabulary = Dictionary(questions)
            self.vocabulary.save(path)
        else:
            self.vocabulary = Dictionary.load(path)
        self.w_C = compute_w_C(questions, self.vocabulary)  # background lm
        self.model = TRLM([], self.w_C, self.alignments, len(self.vocabulary), alpha=self.alpha, sigma=self.sigma)

        del self.trainset


    def set_parameters(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
        self.model = TRLM([], self.w_C, self.alignments, len(self.vocabulary), alpha=self.alpha, sigma=self.sigma)


    def validate(self):
        scores = {}
        for j, pair in enumerate(self.devset):
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            q1id = pair['qid1']
            q1 = pair['tokens_proc1'] if self.stop else pair['tokens1']
            if self.stop:
                q1emb = self.encode(q1id, q1, self.devidx, self.develmo)
            else:
                q1emb = self.encode(q1id, q1, self.fulldevidx, self.fulldevelmo)

            q2id = pair['qid2']
            q2 = pair['tokens_proc2'] if self.stop else pair['tokens2']
            if self.stop:
                q2emb = self.encode(q2id, q2, self.devidx, self.develmo)
            else:
                q2emb = self.encode(q2id, q2, self.fulldevidx, self.fulldevelmo)

            if self.vector == 'alignments':
                lmprob, trmprob, score, _ = self.model.score(q1, q2)
            else:
                lmprob, trmprob, score, _ = self.model(q1, q1emb, q2, q2emb)

        return scores