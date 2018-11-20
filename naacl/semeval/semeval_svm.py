__author__='thiagocastroferreira'

import sys
sys.path.append('../')

import _pickle as p
import os
import numpy as np

from models.svm import Model
from semeval import Semeval
from semeval_bm25 import SemevalBM25
from semeval_cosine import SemevalCosine, SemevalSoftCosine
from semeval_translation import SemevalTranslation

from sklearn.preprocessing import MinMaxScaler

DATA_PATH='data'
FEATURES_PATH = os.path.join(DATA_PATH, 'trainfeatures.pickle')

class SemevalSVM(Semeval):
    def __init__(self, features='bm25,', comment_features='bm25,', vector='word2vec', path=FEATURES_PATH):
        Semeval.__init__(self)
        self.path = path
        self.features = features.split(',')
        self.comment_features = comment_features.split(',')
        self.vector = vector
        self.svm = Model()

        self.bm25 = SemevalBM25()
        self.cosine = SemevalCosine()
        self.softcosine = SemevalSoftCosine(vector)
        self.translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector=self.vector)

        self.train()


    def train(self):
        if not os.path.exists(self.path):
            X, y = [], []
            for i, query_question in enumerate(self.traindata):
                percentage = round(float(i + 1) / len(self.traindata), 2)
                print('Preparing traindata: ', percentage, i + 1, sep='\t', end = '\r')
                q1id = query_question['q1_id']
                q2id = query_question['q2_id']
                q1, q2 = query_question['q1'], query_question['q2']
                x = []

                q1_emb = self.encode(q1id, q1, self.trainidx, self.trainelmo, self.vector)
                q2_emb = self.encode(q2id, q2, self.trainidx, self.trainelmo, self.vector)

                # bm25
                if 'bm25' in self.features:
                    score = self.bm25.model(q1, q2id)
                    x.append(score)

                # translation
                if 'translation' in self.features:
                    if self.vector == 'alignments':
                        lmprob, trmprob, trlmprob, proctime = self.translation.model.score(q1, q2)
                    else:
                        lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q2, q2_emb)
                    x.append(trlmprob)

                # cosine
                if 'cosine' in self.features:
                    score = self.cosine.model(q1, q2)
                    x.append(score)

                # softcosine
                if 'softcosine' in self.features:
                    if self.vector == 'alignments':
                        score = self.softcosine.model.score(q1, q2, self.alignments)
                    else:
                        score = self.softcosine.model(q1, q1_emb, q2, q2_emb)
                    x.append(score)

                for comment in query_question['comments']:
                    q3id = comment['id']
                    q3 = comment['tokens']

                    if len(q3) > 0:
                        q3_emb = self.encode(q3id, q3, self.trainidx, self.trainelmo, self.vector)
                        # bm25
                        if 'bm25' in self.comment_features:
                            score = self.bm25.model(q1, q3id)
                            x.append(score)

                        # translation
                        if 'translation' in self.comment_features:
                            if self.vector == 'alignments':
                                lmprob, trmprob, trlmprob, proctime = self.translation.model.score(q1, q3)
                            else:
                                lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q3, q3_emb)
                            x.append(trlmprob)

                        # cosine
                        if 'cosine' in self.comment_features:
                            score = self.cosine.model(q1, q3)
                            x.append(score)

                        # softcosine
                        if 'softcosine' in self.comment_features:
                            if self.vector == 'alignments':
                                score = self.softcosine.model.score(q1, q2, self.alignments)
                            else:
                                score = self.softcosine.model(q1, q1_emb, q3, q3_emb)
                            x.append(score)
                    else:
                        # bm25
                        if 'bm25' in self.comment_features:
                            x.append(0)

                        # translation
                        if 'translation' in self.comment_features:
                            x.append(0)

                        # cosine
                        if 'cosine' in self.comment_features:
                            x.append(0)

                        # softcosine
                        if 'softcosine' in self.comment_features:
                            x.append(0)

                X.append(x)
                y.append(query_question['label'])

            p.dump(list(zip(X, y)), open(self.path, 'wb'))
        else:
            f = p.load(open(self.path, 'rb'))
            X = np.array([x[0] for x in f])
            y = list(map(lambda x: x[1], f))

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.model = self.svm.train_svm(
            trainvectors=X,
            labels=y,
            c='search',
            kernel='search',
            gamma='search',
            jobs=4
        )

    def validate(self):
        ranking = {}
        y_real, y_pred = [], []
        for j, q1id in enumerate(self.devset):
            ranking[q1id] = []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']
            q1_emb = self.encode(q1id, q1, self.devidx, self.develmo, self.vector)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens_proc']
                q2_emb = self.encode(q2id, q2, self.devidx, self.develmo, self.vector)

                x = []
                # bm25
                if 'bm25' in self.features:
                    score = self.bm25.model(q1, q2id)
                    x.append(score)

                # translation
                if 'translation' in self.features:
                    if self.vector == 'alignments':
                        lmprob, trmprob, trlmprob, proctime = self.translation.model.score(q1, q2)
                    else:
                        lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q2, q2_emb)
                    x.append(trlmprob)

                # cosine
                if 'cosine' in self.features:
                    score = self.cosine.model(q1, q2)
                    x.append(score)

                # softcosine
                if 'softcosine' in self.features:
                    if self.vector == 'alignments':
                        score = self.softcosine.model.score(q1, q2, self.alignments)
                    else:
                        score = self.softcosine.model(q1, q1_emb, q2, q2_emb)
                    x.append(score)

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    q3id = rel_comment['id']
                    q3 = rel_comment['tokens_proc']

                    if len(q3) > 0:
                        q3_emb = self.encode(q3id, q3, self.devidx, self.develmo, self.vector)
                        # bm25
                        if 'bm25' in self.comment_features:
                            score = self.bm25.model(q1, q3id)
                            x.append(score)

                        # translation
                        if 'translation' in self.comment_features:
                            if self.vector == 'alignments':
                                lmprob, trmprob, trlmprob, proctime = self.translation.model.score(q1, q3)
                            else:
                                lmprob, trmprob, trlmprob, proctime = self.translation.model(q1, q1_emb, q3, q3_emb)
                            x.append(trlmprob)

                        # cosine
                        if 'cosine' in self.comment_features:
                            score = self.cosine.model(q1, q3)
                            x.append(score)

                        # softcosine
                        if 'softcosine' in self.comment_features:
                            if self.vector == 'alignments':
                                score = self.softcosine.model.score(q1, q2, self.alignments)
                            else:
                                score = self.softcosine.model(q1, q1_emb, q3, q3_emb)
                            x.append(score)
                    else:
                        # bm25
                        if 'bm25' in self.comment_features:
                            x.append(0)

                        # translation
                        if 'translation' in self.comment_features:
                            x.append(0)

                        # cosine
                        if 'cosine' in self.comment_features:
                            x.append(0)

                        # softcosine
                        if 'softcosine' in self.comment_features:
                            x.append(0)

                x = self.scaler.transform([x])
                score, pred_label = self.model.score(x)
                y_pred.append(pred_label)

                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                y_real.append(real_label)
                ranking[q1id].append((real_label, score, q2id))

                parameter_settings = self.model.return_parameter_settings(clf='svm')
                
        return ranking, y_real, y_pred, parameter_settings
