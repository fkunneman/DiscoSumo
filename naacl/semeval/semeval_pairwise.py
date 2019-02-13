__author__='thiagocastroferreira'

import sys
sys.path.append('../')
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import _pickle as p
import copy
import os
import utils

from semeval import Semeval
from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalSoftCosine

from models.svm import Model
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import f1_score, accuracy_score

class SemevalPairwise(Semeval):
    def __init__(self, stop=True, lowercase=True, punctuation=True, vector='word2vec', w2vdim=300, proctrain=True):
        Semeval.__init__(self, stop=stop, vector=vector, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain, w2vdim=w2vdim)
        self.test2016pairdata, _, _, _ = self.format_pairwise_data(self.testset2016)

        self.test2017pairdata, _, _, _ = self.format_pairwise_data(self.testset2017)

        self.devpairdata, _, _, _ = self.format_pairwise_data(self.devset)

        self.trainpairdata, self.voc2id, self.id2voc, self.vocabulary = self.format_pairwise_data(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.trainpairdata))

        self.ensemble = Model()

        self.train()


    def format(self, ranking):
        new_ranking = {}
        for q1id in ranking:
            new_ranking[q1id] = {}
            for question in ranking[q1id]:
                real_label, score, q2id = question
                new_ranking[q1id][q2id] = (score, real_label)
        return new_ranking


    def train_bm25(self):
        path = os.path.join('pairwise', 'bm25.lower_' + str(True) + '.stop_' + str(False) + '.punct_' + str(True))
        if not os.path.exists(path):
            self.bm25 = SemevalBM25(stop=True, lowercase=False, punctuation=True, proctrain=True)
            self.trainbm25 = self.format(self.bm25.test(self.bm25.traindata))
            self.trainpairbm25 = self.bm25.pairs(self.bm25.traindata)
            self.traincommentbm25 = self.bm25.comments(self.bm25.traindata)

            self.devbm25 = self.format(self.bm25.validate())
            self.devpairbm25 = self.bm25.pairs(self.bm25.devdata)
            self.devcommentbm25 = self.bm25.comments(self.bm25.devdata)

            self.test2016bm25 = self.format(self.bm25.test(self.bm25.test2016data))
            self.test2016pairbm25 = self.bm25.pairs(self.bm25.test2016data)
            self.test2016commentbm25 = self.bm25.comments(self.bm25.test2016data)

            self.test2017bm25 = self.format(self.bm25.test(self.bm25.test2017data))
            self.test2017pairbm25 = self.bm25.pairs(self.bm25.test2017data)
            self.test2017commentbm25 = self.bm25.comments(self.bm25.test2017data)
            del self.bm25

            data = {
                'train': self.trainbm25, 'trainpair': self.trainpairbm25, 'traincomment': self.traincommentbm25,
                'dev': self.devbm25, 'devpair': self.devpairbm25, 'devcomment': self.devcommentbm25,
                'test2016': self.test2016bm25, 'test2016pair': self.test2016pairbm25, 'test2016comment': self.test2016commentbm25,
                'test2017':self.test2017bm25, 'test2017pair': self.test2017pairbm25, 'test2017comment': self.test2017commentbm25,
            }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainbm25 = data['train']
            self.trainpairbm25 = data['trainpair']
            self.traincommentbm25 = data['traincomment']

            self.devbm25 = data['dev']
            self.devpairbm25 = data['devpair']
            self.devcommentbm25 = data['devcomment']

            self.test2016bm25 = data['test2016']
            self.test2016pairbm25 = data['test2016pair']
            self.test2016commentbm25 = data['test2016comment']

            self.test2017bm25 = data['test2017']
            self.test2017pairbm25 = data['test2017pair']
            self.test2017commentbm25 = data['test2017comment']


    def train_translation(self):
        path = os.path.join('pairwise', 'translation.lower_' + str(False) + '.stop_' + str(True) + '.punct_' + str(True) + '.vector_' + str('word2vec') + '.vecdim_' + str(self.w2vdim))
        if not os.path.exists(path):
            self.translation = SemevalTranslation(alpha=0.8, sigma=0.2, punctuation=True, proctrain=True, vector='word2vec', stop=True, lowercase=False, w2vdim=self.w2vdim)
            self.traintranslation = self.format(self.translation.test(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo))
            self.trainpairtranslation = self.translation.pairs(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo)
            self.traincommenttranslation = self.translation.comments(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo)

            self.devtranslation = self.format(self.translation.validate())
            self.devpairtranslation = self.translation.pairs(self.translation.devdata, self.translation.devidx, self.translation.develmo)
            self.devcommenttranslation = self.translation.comments(self.translation.devdata, self.translation.devidx, self.translation.develmo)

            self.test2016translation = self.format(self.translation.test(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo))
            self.test2016pairtranslation = self.translation.pairs(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo)
            self.test2016commenttranslation = self.translation.comments(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo)

            self.test2017translation = self.format(self.translation.test(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo))
            self.test2017pairtranslation = self.translation.pairs(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo)
            self.test2017commenttranslation = self.translation.comments(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo)
            del self.translation

            data = {
                'train': self.traintranslation, 'trainpair': self.trainpairtranslation,
                'traincomment': self.traincommenttranslation,
                'dev': self.devtranslation, 'devpair': self.devpairtranslation,
                'devcomment': self.devcommenttranslation,
                'test2016': self.test2016translation, 'test2016pair': self.test2016pairtranslation,
                'test2016comment': self.test2016commenttranslation,
                'test2017':self.test2017translation, 'test2017pair': self.test2017pairtranslation,
                'test2017comment': self.test2017commenttranslation
            }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.traintranslation = data['train']
            self.trainpairtranslation = data['trainpair']
            self.traincommenttranslation = data['traincomment']

            self.devtranslation = data['dev']
            self.devpairtranslation = data['devpair']
            self.devcommenttranslation = data['devcomment']

            self.test2016translation = data['test2016']
            self.test2016pairtranslation = data['test2016pair']
            self.test2016commenttranslation = data['test2016comment']

            self.test2017translation = data['test2017']
            self.test2017pairtranslation = data['test2017pair']
            self.test2017commenttranslation = data['test2017comment']


    def train_softcosine(self):
        path = os.path.join('pairwise', 'softcosine.lower_' + str(True) + '.stop_' + str(True) + '.punct_' + str(True) + '.vector_' + str('word2vec+elmo') + '.vecdim_' + str(self.w2vdim))
        if not os.path.exists(path):
            self.softcosine = SemevalSoftCosine(stop=True, vector='word2vec+elmo', lowercase=True, punctuation=True, proctrain=True, w2vdim=self.w2vdim)

            self.trainsoftcosine = self.format(self.softcosine.test(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo))
            self.trainpairsoftcosine = self.softcosine.pairs(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo)
            self.traincommentsoftcosine = self.softcosine.comments(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo)

            self.devsoftcosine = self.format(self.softcosine.validate())
            self.devpairsoftcosine = self.softcosine.pairs(self.softcosine.devdata, self.softcosine.devidx, self.softcosine.develmo)
            self.devcommentsoftcosine = self.softcosine.comments(self.softcosine.devdata, self.softcosine.devidx, self.softcosine.develmo)

            self.test2016softcosine = self.format(self.softcosine.test(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo))
            self.test2016pairsoftcosine = self.softcosine.pairs(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo)
            self.test2016commentsoftcosine = self.softcosine.comments(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo)

            self.test2017softcosine = self.format(self.softcosine.test(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo))
            self.test2017pairsoftcosine = self.softcosine.pairs(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo)
            self.test2017commentsoftcosine = self.softcosine.comments(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo)
            del self.softcosine

            data = {
                'train': self.trainsoftcosine, 'trainpair': self.trainpairsoftcosine,
                'traincomment': self.traincommentsoftcosine,
                'dev': self.devsoftcosine, 'devpair': self.devpairsoftcosine,
                'devcomment': self.devcommentsoftcosine,
                'test2016': self.test2016softcosine, 'test2016pair': self.test2016pairsoftcosine,
                'test2016comment': self.test2016commentsoftcosine,
                'test2017':self.test2017softcosine, 'test2017pair': self.test2017pairsoftcosine,
                'test2017comment': self.test2017commentsoftcosine
            }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainsoftcosine = data['train']
            self.trainpairsoftcosine = data['trainpair']
            self.traincommentsoftcosine = data['traincomment']

            self.devsoftcosine = data['dev']
            self.devpairsoftcosine = data['devpair']
            self.devcommentsoftcosine = data['devcomment']

            self.test2016softcosine = data['test2016']
            self.test2016pairsoftcosine = data['test2016pair']
            self.test2016commentsoftcosine = data['test2016comment']

            self.test2017softcosine = data['test2017']
            self.test2017pairsoftcosine = data['test2017pair']
            self.test2017commentsoftcosine = data['test2017comment']


    def get_features(self, q1id, q2id, q3id, bm25, bm25comment, translation, translationcomment, softcosine, softcosinecomment):
        bm25q1q2 = bm25[q1id][q2id][0]
        translationq1q2 = translation[q1id][q2id][0]
        softcosineq1q2 = softcosine[q1id][q2id][0]

        bm25commentq1q2 = bm25comment[q1id][q2id]
        translationcommentq1q2 = translationcomment[q1id][q2id]
        softcosinecommentq1q2 = softcosinecomment[q1id][q2id]

        bm25q1q3 = bm25[q1id][q3id][0]
        translationq1q3 = translation[q1id][q3id][0]
        softcosineq1q3 = softcosine[q1id][q3id][0]

        bm25commentq1q3 = bm25comment[q1id][q3id]
        translationcommentq1q3 = translationcomment[q1id][q3id]
        softcosinecommentq1q3 = softcosinecomment[q1id][q3id]

        X = []
        X.append(bm25q1q2)
        X.append(translationq1q2)
        X.append(softcosineq1q2)

        X.extend(bm25commentq1q2)
        X.extend(translationcommentq1q2)
        X.extend(softcosinecommentq1q2)

        X.append(bm25q1q3)
        X.append(translationq1q3)
        X.append(softcosineq1q3)

        X.extend(bm25commentq1q3)
        X.extend(translationcommentq1q3)
        X.extend(softcosinecommentq1q3)
        return X


    def train(self):
        print('Initializing BM25...')
        self.train_bm25()

        print('Initializing Translation...')
        self.train_translation()

        print('Initializing Softcosine...')
        self.train_softcosine()

        self.X, self.y = [], []
        for q1id in self.trainpairdata:
            for pair in self.trainpairdata[q1id]:
                q2id = pair['q2_id']
                q3id = pair['q3_id']

                X = self.get_features(q1id, q2id, q3id,
                                      self.trainbm25, self.traincommentbm25,
                                      self.traintranslation, self.traincommenttranslation,
                                      self.trainsoftcosine, self.traincommentsoftcosine)
                self.X.append(X)

                self.y.append(pair['label'])

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        self.ensemble.train_regression(trainvectors=self.X, labels=self.y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def validate(self, procdata, set_='dev'):
        if set_ == 'dev':
            bm25 = self.devbm25
            bm25comment = self.devcommentbm25

            translation = self.devtranslation
            translationcomment = self.devcommenttranslation

            softcosine = self.devsoftcosine
            softcosinecomment = self.devcommentsoftcosine
        elif set_ == 'test2016':
            bm25 = self.test2016bm25
            bm25comment = self.test2016commentbm25

            translation = self.test2016translation
            translationcomment = self.test2016commenttranslation

            softcosine = self.test2016softcosine
            softcosinecomment = self.test2016commentsoftcosine
        else:
            bm25 = self.test2017bm25
            bm25comment = self.test2017commentbm25

            translation = self.test2017translation
            translationcomment = self.test2017commenttranslation

            softcosine = self.test2017softcosine
            softcosinecomment = self.test2017commentsoftcosine

        y_real, y_pred = [], []
        for q1id in procdata:
            for pair in procdata[q1id]:
                q2id = pair['q2_id']
                q3id = pair['q3_id']

                X = self.get_features(q1id, q2id, q3id,
                                      bm25, bm25comment,
                                      translation, translationcomment,
                                      softcosine, softcosinecomment)

                X = self.scaler.transform([X])[0]
                clfscore, pred_label = self.ensemble.score(X)
                y_pred.append(pred_label)
                y_real.append(pair['label'])

        f1score = f1_score(y_real, y_pred)
        accuracy = accuracy_score(y_real, y_pred)

        print('Accuracy: ', accuracy)
        print('F-Score: ', f1score)


    def test(self, set_='dev'):
        if set_ == 'train':
            procdata = self.traindata

            bm25 = self.trainbm25
            bm25comment = self.traincommentbm25

            translation = self.traintranslation
            translationcomment = self.traincommenttranslation

            softcosine = self.trainsoftcosine
            softcosinecomment = self.traincommentsoftcosine
        elif set_ == 'dev':
            procdata = self.devdata

            bm25 = self.devbm25
            bm25comment = self.devcommentbm25

            translation = self.devtranslation
            translationcomment = self.devcommenttranslation

            softcosine = self.devsoftcosine
            softcosinecomment = self.devcommentsoftcosine
        elif set_ == 'test2016':
            procdata = self.test2016data

            bm25 = self.test2016bm25
            bm25comment = self.test2016commentbm25

            translation = self.test2016translation
            translationcomment = self.test2016commenttranslation

            softcosine = self.test2016softcosine
            softcosinecomment = self.test2016commentsoftcosine
        else:
            procdata = self.test2017data

            bm25 = self.test2017bm25
            bm25comment = self.test2017commentbm25

            translation = self.test2017translation
            translationcomment = self.test2017commenttranslation

            softcosine = self.test2017softcosine
            softcosinecomment = self.test2017commentsoftcosine

        ranking = {}
        for q1id in procdata:
            ranking[q1id] = []
            question_ids = []
            for q2id in procdata[q1id]:
                question_ids.append((q2id, procdata[q1id][q2id]['ranking']))
            question_ids = [w[0] for w in sorted(question_ids, key=lambda x: x[1])]

            questions = {}
            pairs = {}
            for q2id in procdata[q1id]:
                pairs[q2id] = {}
                questions[q2id] = {
                    'bm25': bm25[q1id][q2id][0],
                    'translation': translation[q1id][q2id][0],
                    'softcosine': softcosine[q1id][q2id][0],
                    'bm25comment': bm25comment[q1id][q2id],
                    'translationcomment': translationcomment[q1id][q2id],
                    'softcosinecomment': softcosinecomment[q1id][q2id],
                    'q1_full': procdata[q1id][q2id]['q1_full'],
                    'q2_full': procdata[q1id][q2id]['q2_full'],
                }


            qids = self.sort(question_ids, questions)
            for i, q2id in enumerate(qids):
                ranking[q1id].append((1, 10-i, q2id))
        return ranking


    def sort(self, question_ids, questions):
        if len(question_ids) <= 1:
            return question_ids

        half = int(len(question_ids) / 2)
        group1 = self.sort(question_ids[:half], questions)
        group2 = self.sort(question_ids[half:], questions)

        result = []
        i1, i2 = 0, 0
        while i1 < len(group1) or i2 < len(group2):
            if i1 == len(group1):
                result.append(group2[i2])
                i2 += 1
            elif i2 == len(group2):
                result.append(group1[i1])
                i1 += 1
            else:
                q2id, q3id = group1[i1], group2[i2]

                bm25q1q2 = questions[q2id]['bm25']
                translationq1q2 = questions[q2id]['translation']
                softcosineq1q2 = questions[q2id]['softcosine']

                bm25commentq1q2 = questions[q2id]['bm25comment']
                translationcommentq1q2 = questions[q2id]['translationcomment']
                softcosinecommentq1q2 = questions[q2id]['softcosinecomment']

                bm25q1q3 = questions[q3id]['bm25']
                translationq1q3 = questions[q3id]['translation']
                softcosineq1q3 = questions[q3id]['softcosine']

                bm25commentq1q3 = questions[q3id]['bm25comment']
                translationcommentq1q3 = questions[q3id]['translationcomment']
                softcosinecommentq1q3 = questions[q3id]['softcosinecomment']

                X = []
                X.append(bm25q1q2)
                X.append(translationq1q2)
                X.append(softcosineq1q2)

                X.extend(bm25commentq1q2)
                X.extend(translationcommentq1q2)
                X.extend(softcosinecommentq1q2)

                X.append(bm25q1q3)
                X.append(translationq1q3)
                X.append(softcosineq1q3)

                X.extend(bm25commentq1q3)
                X.extend(translationcommentq1q3)
                X.extend(softcosinecommentq1q3)

                X = self.scaler.transform([X])[0]
                clfscore, pred_label = self.ensemble.score(X)

                if pred_label == 1:
                    result.append(q3id)
                    i2 += 1
                else:
                    result.append(q2id)
                    i1 += 1
        return result


if __name__ == '__main__':
    semeval = SemevalPairwise()

    semeval.validate(semeval.devpairdata, 'dev')

    ranking = semeval.test('dev')
    map_baseline, map_model = utils.evaluate(copy.copy(ranking), utils.prepare_gold(utils.DEV_GOLD_PATH))

    print('Dev:')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)

    ranking = semeval.test('test2016')
    map_baseline, map_model = utils.evaluate(copy.copy(ranking), utils.prepare_gold(utils.TEST2016_GOLD_PATH))
    print('\nTest2016:')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)

    ranking = semeval.test('test2017')
    map_baseline, map_model = utils.evaluate(copy.copy(ranking), utils.prepare_gold(utils.TEST2017_GOLD_PATH))
    print('\nTest2017:')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)