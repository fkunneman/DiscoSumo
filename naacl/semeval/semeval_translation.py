__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import copy
import os
from semeval import Semeval
from gensim.corpora import Dictionary
from models.translation import TRLM, compute_w_C

DATA_PATH='data'
TRANSLATION_PATH='alignments/model/lex.f2e'

class SemevalTranslation(Semeval):
    def __init__(self, alpha, sigma, stop=True, lowercase=True, vector='word2vec'):
        Semeval.__init__(self, stop=stop, vector=vector, lowercase=lowercase)
        self.alpha = alpha
        self.sigma = sigma
        self.train()

    def train(self):
        questions = copy.copy(self.additional)
        for i, q1id in enumerate(self.trainset):
            question = self.trainset[q1id]
            q1 = [w.lower() for w in question['tokens']] if self.lowercase else question['tokens']
            questions.append(q1)

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                questions.append(q2)

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    q3 = [w.lower() for w in rel_comment['tokens']] if self.lowercase else rel_comment['tokens']
                    questions.append(q3)

        path = os.path.join(DATA_PATH, 'transdict.model')
        if not os.path.exists(path):
            self.vocabulary = Dictionary(questions)
            self.vocabulary.save(path)
        else:
            self.vocabulary = Dictionary.load(path)
        self.w_C = compute_w_C(questions, self.vocabulary)  # background lm
        self.model = TRLM([], self.w_C, self.alignments, len(self.vocabulary), alpha=self.alpha, sigma=self.sigma)

        del self.additional
        del self.trainset
        del self.traindata


    def set_parameters(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
        self.model = TRLM([], self.w_C, self.alignments, len(self.vocabulary), alpha=self.alpha, sigma=self.sigma)


    def validate(self):
        ranking = {}
        for j, q1id in enumerate(self.devdata):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devdata), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            for q2id in self.devdata[q1id]:
                pair = self.devdata[q1id][q2id]
                q1, q2 = pair['q1'], pair['q2']

                if self.stop:
                    q1emb = self.encode(q1id, q1, self.devidx, self.develmo)
                    q2emb = self.encode(q2id, q2, self.devidx, self.develmo)
                else:
                    q1emb = self.encode(q1id, q1, self.fulldevidx, self.fulldevelmo)
                    q2emb = self.encode(q2id, q2, self.fulldevidx, self.fulldevelmo)

                if self.vector == 'alignments':
                    lmprob, trmprob, score, _ = self.model.score(q1, q2)
                else:
                    lmprob, trmprob, score, _ = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking


    def test(self, testdata, elmoidx, elmovec, fullelmoidx, fullelmovec):
        self.testdata = testdata
        ranking = {}
        for j, q1id in enumerate(self.testdata):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.testdata), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            for q2id in self.testdata[q1id]:
                pair = self.testdata[q1id][q2id]
                q1, q2 = pair['q1'], pair['q2']

                if self.stop:
                    q1emb = self.encode(q1id, q1, elmoidx, elmovec)
                    q2emb = self.encode(q2id, q2, elmoidx, elmovec)
                else:
                    q1emb = self.encode(q1id, q1, fullelmoidx, fullelmovec)
                    q2emb = self.encode(q2id, q2, fullelmoidx, fullelmovec)

                if self.vector == 'alignments':
                    lmprob, trmprob, score, _ = self.model.score(q1, q2)
                else:
                    lmprob, trmprob, score, _ = self.model(q1, q1emb, q2, q2emb)

                real_label = pair['label']
                ranking[q1id].append((real_label, score, q2id))
        return ranking