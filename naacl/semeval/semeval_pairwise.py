__author__='thiagocastroferreira'

import sys
sys.path.append('../')
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import ev, metrics
import _pickle as p
import copy
import os
import paths

from semeval import Semeval
from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalSoftCosine

from operator import itemgetter
from models.svm import Model
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import f1_score, accuracy_score

DEV_GOLD_PATH=paths.DEV_GOLD_PATH
TEST2016_GOLD_PATH=paths.TEST2016_GOLD_PATH
TEST2017_GOLD_PATH=paths.TEST2017_GOLD_PATH

def prepare_gold(path):
    ir = ev.read_res_file_aid(path, 'trec')
    return ir


def evaluate(ranking, gold):
    for qid in gold:
        gold_sorted = sorted(gold[qid], key = itemgetter(2), reverse = True)
        pred_sorted = ranking[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(2), reverse = True)

        gold[qid], ranking[qid] = [], []
        for i, row in enumerate(gold_sorted):
            relevant, gold_score, aid = row
            gold[qid].append((relevant, gold_score, aid))

            pred_score = pred_sorted[i][1]
            ranking[qid].append((relevant, pred_score, aid))

    for qid in gold:
        # Sort by IR score.
        gold_sorted = sorted(gold[qid], key = itemgetter(1), reverse = True)

        # Sort by SVM prediction score.
        pred_sorted = ranking[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(1), reverse = True)

        gold[qid] = [rel for rel, score, aid in gold_sorted]
        ranking[qid] = [rel for rel, score, aid in pred_sorted]

    map_gold = metrics.map(gold, 10)
    map_pred = metrics.map(ranking, 10)
    return map_gold, map_pred

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

            self.devbm25 = self.format(self.bm25.validate())
            self.devpairbm25 = self.bm25.pairs(self.bm25.devdata)

            self.test2016bm25 = self.format(self.bm25.test(self.bm25.test2016data))
            self.test2016pairbm25 = self.bm25.pairs(self.bm25.test2016data)

            self.test2017bm25 = self.format(self.bm25.test(self.bm25.test2017data))
            self.test2017pairbm25 = self.bm25.pairs(self.bm25.test2017data)
            del self.bm25

            data = {
                'train': self.trainbm25, 'trainpair': self.trainpairbm25,
                'dev': self.devbm25, 'devpair': self.devpairbm25,
                'test2016': self.test2016bm25, 'test2016pair': self.test2016pairbm25,
                'test2017':self.test2017bm25, 'test2017pair': self.test2017pairbm25
            }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainbm25 = data['train']
            self.trainpairbm25 = data['trainpair']

            self.devbm25 = data['dev']
            self.devpairbm25 = data['devpair']

            self.test2016bm25 = data['test2016']
            self.test2016pairbm25 = data['test2016pair']

            self.test2017bm25 = data['test2017']
            self.test2017pairbm25 = data['test2017pair']


    def train_translation(self):
        path = os.path.join('pairwise', 'translation.lower_' + str(False) + '.stop_' + str(True) + '.punct_' + str(True) + '.vector_' + str('word2vec') + '.vecdim_' + str(self.w2vdim))
        if not os.path.exists(path):
            self.translation = SemevalTranslation(alpha=0.8, sigma=0.2, punctuation=True, proctrain=True, vector='word2vec', stop=True, lowercase=False, w2vdim=self.w2vdim)
            self.traintranslation = self.format(self.translation.test(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo))
            self.trainpairtranslation = self.translation.pairs(self.translation.traindata, self.translation.trainidx, self.translation.trainelmo)

            self.devtranslation = self.format(self.translation.validate())
            self.devpairtranslation = self.translation.pairs(self.translation.devdata, self.translation.devidx, self.translation.develmo)

            self.test2016translation = self.format(self.translation.test(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo))
            self.test2016pairtranslation = self.translation.pairs(self.translation.test2016data, self.translation.test2016idx, self.translation.test2016elmo)

            self.test2017translation = self.format(self.translation.test(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo))
            self.test2017pairtranslation = self.translation.pairs(self.translation.test2017data, self.translation.test2017idx, self.translation.test2017elmo)
            del self.translation

            data = {
                'train': self.traintranslation, 'trainpair': self.trainpairtranslation,
                'dev': self.devtranslation, 'devpair': self.devpairtranslation,
                'test2016': self.test2016translation, 'test2016pair': self.test2016pairtranslation,
                'test2017':self.test2017translation, 'test2017pair': self.test2017pairtranslation
            }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.traintranslation = data['train']
            self.trainpairtranslation = data['trainpair']

            self.devtranslation = data['dev']
            self.devpairtranslation = data['devpair']

            self.test2016translation = data['test2016']
            self.test2016pairtranslation = data['test2016pair']

            self.test2017translation = data['test2017']
            self.test2017pairtranslation = data['test2017pair']


    def train_softcosine(self):
        path = os.path.join('pairwise', 'softcosine.lower_' + str(True) + '.stop_' + str(True) + '.punct_' + str(True) + '.vector_' + str('word2vec+elmo') + '.vecdim_' + str(self.w2vdim))
        if not os.path.exists(path):
            self.softcosine = SemevalSoftCosine(stop=True, vector='word2vec+elmo', lowercase=True, punctuation=True, proctrain=True, w2vdim=self.w2vdim)

            self.trainsoftcosine = self.format(self.softcosine.test(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo))
            self.trainpairsoftcosine = self.softcosine.pairs(self.softcosine.traindata, self.softcosine.trainidx, self.softcosine.trainelmo)

            self.devsoftcosine = self.format(self.softcosine.validate())
            self.devpairsoftcosine = self.softcosine.pairs(self.softcosine.devdata, self.softcosine.devidx, self.softcosine.develmo)

            self.test2016softcosine = self.format(self.softcosine.test(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo))
            self.test2016pairsoftcosine = self.softcosine.pairs(self.softcosine.test2016data, self.softcosine.test2016idx, self.softcosine.test2016elmo)

            self.test2017softcosine = self.format(self.softcosine.test(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo))
            self.test2017pairsoftcosine = self.softcosine.pairs(self.softcosine.test2017data, self.softcosine.test2017idx, self.softcosine.test2017elmo)
            del self.softcosine

            data = {
                'train': self.trainsoftcosine, 'trainpair': self.trainpairsoftcosine,
                'dev': self.devsoftcosine, 'devpair': self.devpairsoftcosine,
                'test2016': self.test2016softcosine, 'test2016pair': self.test2016pairsoftcosine,
                'test2017':self.test2017softcosine, 'test2017pair': self.test2017pairsoftcosine
            }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainsoftcosine = data['train']
            self.trainpairsoftcosine = data['trainpair']

            self.devsoftcosine = data['dev']
            self.devpairsoftcosine = data['devpair']

            self.test2016softcosine = data['test2016']
            self.test2016pairsoftcosine = data['test2016pair']

            self.test2017softcosine = data['test2017']
            self.test2017pairsoftcosine = data['test2017pair']


    def get_features(self, q1id, q2id, q3id, bm25, bm25pair, translation, translationpair, softcosine, softcosinepair):
        bm25q1q2 = bm25[q1id][q2id][0]
        translationq1q2 = translation[q1id][q2id][0]
        softcosineq1q2 = softcosine[q1id][q2id][0]

        bm25q1q3 = bm25[q1id][q3id][0]
        translationq1q3 = translation[q1id][q3id][0]
        softcosineq1q3 = softcosine[q1id][q3id][0]

        bm25q2q3 = bm25pair[q1id][q2id][q3id]
        translationq2q3 = translationpair[q1id][q2id][q3id]
        softcosineq2q3 = softcosinepair[q1id][q2id][q3id]

        X = []
        X.append(bm25q1q2)
        X.append(translationq1q2)
        X.append(softcosineq1q2)
        X.append(bm25q1q3)
        X.append(translationq1q3)
        X.append(softcosineq1q3)
        X.append(bm25q2q3)
        X.append(translationq2q3)
        X.append(softcosineq2q3)
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
                                      self.trainbm25, self.trainpairbm25,
                                      self.traintranslation, self.trainpairtranslation,
                                      self.trainsoftcosine, self.trainpairsoftcosine)
                self.X.append(X)

                self.y.append(pair['label'])

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        self.ensemble.train_regression(trainvectors=self.X, labels=self.y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def validate(self, procdata, set_='dev'):
        if set_ == 'dev':
            bm25 = self.devbm25
            bm25pair = self.devpairbm25

            translation = self.devtranslation
            translationpair = self.devpairtranslation

            softcosine = self.devsoftcosine
            softcosinepair = self.devpairsoftcosine
        elif set_ == 'test2016':
            bm25 = self.test2016bm25
            bm25pair = self.test2016pairbm25

            translation = self.test2016translation
            translationpair = self.test2016pairtranslation

            softcosine = self.test2016softcosine
            softcosinepair = self.test2016pairsoftcosine
        else:
            bm25 = self.test2017bm25
            bm25pair = self.test2017pairbm25

            translation = self.test2017translation
            translationpair = self.test2017pairtranslation

            softcosine = self.test2017softcosine
            softcosinepair = self.test2017pairsoftcosine

        y_real, y_pred = [], []
        for q1id in procdata:
            for pair in procdata[q1id]:
                q2id = pair['q2_id']
                q3id = pair['q3_id']

                X = self.get_features(q1id, q2id, q3id,
                                      bm25, bm25pair,
                                      translation, translationpair,
                                      softcosine, softcosinepair)

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
            bm25pair = self.trainpairbm25

            translation = self.traintranslation
            translationpair = self.trainpairtranslation

            softcosine = self.trainsoftcosine
            softcosinepair = self.trainpairsoftcosine
        elif set_ == 'dev':
            procdata = self.devdata

            bm25 = self.devbm25
            bm25pair = self.devpairbm25

            translation = self.devtranslation
            translationpair = self.devpairtranslation

            softcosine = self.devsoftcosine
            softcosinepair = self.devpairsoftcosine
        elif set_ == 'test2016':
            procdata = self.test2016data

            bm25 = self.test2016bm25
            bm25pair = self.test2016pairbm25

            translation = self.test2016translation
            translationpair = self.test2016pairtranslation

            softcosine = self.test2016softcosine
            softcosinepair = self.test2016pairsoftcosine
        else:
            procdata = self.test2017data

            bm25 = self.test2017bm25
            bm25pair = self.test2017pairbm25

            translation = self.test2017translation
            translationpair = self.test2017pairtranslation

            softcosine = self.test2017softcosine
            softcosinepair = self.test2017pairsoftcosine

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
                    'q1_full': procdata[q1id][q2id]['q1_full'],
                    'q2_full': procdata[q1id][q2id]['q2_full'],
                }

                for q3id in procdata[q1id]:
                    pairs[q2id][q3id] = {
                        'bm25': bm25pair[q1id][q2id][q3id],
                        'translation': translationpair[q1id][q2id][q3id],
                        'softcosine': softcosinepair[q1id][q2id][q3id],
                    }


            qids = self.sort(question_ids, questions, pairs)
            for i, q2id in enumerate(qids):
                ranking[q1id].append((1, 10-i, q2id))
        return ranking


    def sort(self, question_ids, questions, pairs):
        if len(question_ids) <= 1:
            return question_ids

        half = int(len(question_ids) / 2)
        group1 = self.sort(question_ids[:half], questions, pairs)
        group2 = self.sort(question_ids[half:], questions, pairs)

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

                bm25q1q3 = questions[q3id]['bm25']
                translationq1q3 = questions[q3id]['translation']
                softcosineq1q3 = questions[q3id]['softcosine']

                bm25q2q3 = questions[q2id][q3id]['bm25']
                translationq2q3 = questions[q2id][q3id]['translation']
                softcosineq2q3 = questions[q2id][q3id]['softcosine']

                X = []
                X.append(bm25q1q2)
                X.append(translationq1q2)
                X.append(softcosineq1q2)
                X.append(bm25q1q3)
                X.append(translationq1q3)
                X.append(softcosineq1q3)
                X.append(bm25q2q3)
                X.append(translationq2q3)
                X.append(softcosineq2q3)

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
    map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))

    print('Dev:')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)

    ranking = semeval.test('test2016')
    map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(TEST2016_GOLD_PATH))
    print('\nTest2016:')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)

    ranking = semeval.test('test2017')
    map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(TEST2017_GOLD_PATH))
    print('\nTest2017:')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)