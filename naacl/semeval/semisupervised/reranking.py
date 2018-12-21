__author='thiagocastroferreira'

import sys
sys.path.append('../')
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import ev, metrics
import _pickle as p
import os
import paths
import re

from operator import itemgetter
from semi_bm25 import SemilBM25
from semi_translation import SemiTranslation
from semi_softcosine import SemiSoftCosine

from multiprocessing import Pool

from nltk.corpus import stopwords
stop_ = set(stopwords.words('english'))
from models.svm import Model
from sklearn.preprocessing import MinMaxScaler

DEV_GOLD_PATH=paths.DEV_GOLD_PATH
SEMI_PATH=paths.SEMI_PATH

DATA_PATH='../data'

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


def run(model, testdata):
    return model.test(testdata)

class Rerank:
    def __init__(self, stop={}, lowercase={}, punctuation={}, vector={}, scale=True, alpha=0.9, sigma=0.1):
        self.stop = stop
        self.lowercase = lowercase
        self.punctuation = punctuation
        self.scale = scale
        self.vector = vector
        self.alpha = alpha
        self.sigma = sigma

        self.THREADS = 40

        self.questions, self.ranking = self.load()
        self.ensemble = Model()
        self.train()

        ranking = self.test()
        p.dump(ranking, open(os.path.join(SEMI_PATH, 'reranking'), 'wb'))


    def load(self):
        with open(os.path.join(SEMI_PATH, 'index.txt')) as f:
            indexes = f.read().split('\n')

        with open(os.path.join(SEMI_PATH, 'question.txt')) as f:
            questions = [text.replace('<SENTENCE>', ' ').split() for text in f.read().split('\n')]

        with open(os.path.join(SEMI_PATH, 'ranking')) as f:
            ranking = [w.split() for w in f.read().split('\n')][:-1]

        ranking = dict([(w[0], w[1:]) for w in ranking])
        for qid in ranking:
            ranking[qid] = [w.split('-') for w in ranking[qid]]

        return dict(zip(indexes, questions)), ranking


    def format(self, ranking):
        new_ranking = {}
        for q1id in ranking:
            new_ranking[q1id] = {}
            for question in ranking[q1id]:
                real_label, score, q2id = question
                new_ranking[q1id][q2id] = (score, real_label)
        return new_ranking


    def train(self):
        pool = Pool(processes=self.THREADS)

        lowercase, stop, punctuation = self.lowercase['bm25'], self.stop['bm25'], self.punctuation['bm25']
        path = os.path.join(SEMI_PATH, 'bm25.lower_' + str(lowercase) + '.stop_' + str(stop) + '.punct_' + str(punctuation))
        if not os.path.exists(path):
            self.bm25 = SemilBM25(stop=stop, lowercase=lowercase, punctuation=punctuation)
            self.trainbm25 = self.format(self.bm25.test(self.bm25.traindata))
            self.devbm25 = self.format(self.bm25.validate())
            print('Testing BM25...')
            self.testbm25 = {}
            for q1id in self.ranking:
                self.testbm25[q1id] = {}
                for question in self.ranking[q1id]:
                    q2id, score = question
                    self.testbm25[q1id][q2id] = float(score)
            del self.bm25

            data = {'train': self.trainbm25, 'dev': self.devbm25, 'test': self.testbm25}
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainbm25 = data['train']
            self.devbm25 = data['dev']
            self.testbm25 = data['test']

        vector = self.vector['translation']
        lowercase, stop, punctuation = self.lowercase['translation'], self.stop['translation'], self.punctuation['translation']
        path = os.path.join(SEMI_PATH, 'translation.lower_' + str(lowercase) + '.stop_' + str(stop) + '.punct_' + str(punctuation) + '.vector_' + str(vector))
        if not os.path.exists(path):
            self.translation = SemiTranslation(alpha=self.alpha, sigma=self.sigma, punctuation=punctuation, stop=stop, lowercase=lowercase)
            self.traintranslation = self.format(self.translation.test(self.translation.traindata))
            self.devtranslation = self.format(self.translation.validate())

            print('Testing Translation...')
            testdata = self.format_input(lowercase, stop, punctuation)
            n = int(len(testdata) / self.THREADS)
            chunks = [testdata[i:i+n] for i in range(0, len(testdata), n)]

            processes = []
            for i, chunk in enumerate(chunks):
                print('Process id: ', i+1, 'Doc length:', len(chunk))
                processes.append(pool.apply_async(run, [self.translation, chunk]))

            self.testtranslation = {}
            for i, process in enumerate(processes):
                result = process.get()
                self.testtranslation.update(self.format(result))

            del self.translation

            data = {'train': self.traintranslation, 'dev': self.devtranslation, 'test':self.testtranslation}
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.traintranslation = data['train']
            self.devtranslation = data['dev']
            self.testtranslation = data['test']

        vector = self.vector['softcosine']
        lowercase, stop, punctuation = self.lowercase['softcosine'], self.stop['softcosine'], self.punctuation['softcosine']
        path = os.path.join(SEMI_PATH, 'softcosine.lower_' + str(lowercase) + '.stop_' + str(stop) + '.punct_' + str(punctuation) + '.vector_' + str(vector))
        if not os.path.exists(path):
            self.softcosine = SemiSoftCosine(stop=stop, lowercase=lowercase, punctuation=punctuation)
            self.trainsoftcosine = self.format(self.softcosine.test(self.softcosine.traindata))
            self.devsoftcosine = self.format(self.softcosine.validate())

            print('Testing Softcosine...')
            testdata = self.format_input(lowercase, stop, punctuation)
            n = int(len(testdata) / self.THREADS)
            chunks = [testdata[i:i+n] for i in range(0, len(testdata), n)]

            processes = []
            for i, chunk in enumerate(chunks):
                print('Process id: ', i+1, 'Doc length:', len(chunk))
                processes.append(pool.apply_async(run, [self.softcosine, chunk]))

            self.testsoftcosine = {}
            for i, process in enumerate(processes):
                result = process.get()
                self.testsoftcosine.update(self.format(result))
            del self.softcosine

            data = { 'train': self.trainsoftcosine, 'dev': self.devsoftcosine, 'test': self.testsoftcosine }
            p.dump(data, open(path, 'wb'))
        else:
            data = p.load(open(path, 'rb'))
            self.trainsoftcosine = data['train']
            self.devsoftcosine = data['dev']

        pool.close()
        pool.join()

        self.X, self.y = [], []

        for q1id in self.trainbm25:
            for q2id in self.trainbm25[q1id]:
                X = [self.trainbm25[q1id][q2id][0]]
                X.append(self.traintranslation[q1id][q2id][0])
                X.append(self.trainsoftcosine[q1id][q2id][0])
                self.X.append(X)
                self.y.append(self.trainbm25[q1id][q2id][1])

        if self.scale:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)
        self.ensemble.train_regression(trainvectors=self.X, labels=self.y, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)


    def format_input(self, lowercase, stop, punctuation):
        def remove_punctuation(tokens):
            return re.sub(r'[\W]+',' ', ' '.join(tokens)).strip().split()

        def remove_stopwords(tokens):
            return [w for w in tokens if w.lower() not in stop_]

        procset = {}

        for i, q1id in enumerate(self.ranking):
            procset[q1id] = {}
            percentage = str(round((float(i+1) / len(self.ranking)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')

            q1 = self.questions[q1id]
            q1 = [w.lower() for w in q1] if lowercase else q1
            q1 = remove_punctuation(q1) if punctuation else q1
            q1 = remove_stopwords(q1) if stop else q1

            for row in self.ranking[q1id][1:11]:
                q2id, score = row
                q2 = self.questions[q2id]
                q2 = [w.lower() for w in q2] if lowercase else q2
                q2 = remove_punctuation(q2) if punctuation else q2
                q2 = remove_stopwords(q2) if stop else q2

                label = 0
                procset[q1id][q2id] = {
                    'q1_id': q1id,
                    'q1': q1,
                    'q2_id': q2id,
                    'q2': q2,
                    'label':label
                }

        return procset


    def test(self):
        bm25 = self.testbm25
        translation = self.testtranslation
        softcosine = self.testsoftcosine

        ranking = {}
        for q1id in bm25:
            ranking[q1id] = {}
            for q2id in bm25[q1id]:
                X = [bm25[q1id][q2id][0]]
                X.append(translation[q1id][q2id][0])
                X.append(softcosine[q1id][q2id][0])

                if self.scale:
                    X = self.scaler.transform([X])[0]
                clfscore, pred_label = self.ensemble.score(X)
                ranking[q1id][q2id] = {'score':clfscore, 'label':pred_label}

        return ranking

if __name__ == '__main__':
    lower = {'bm25':True, 'translation':False, 'softcosine':True, 'kernel':False}
    stop = {'bm25':False, 'translation':True, 'softcosine':True, 'kernel':True}
    punctuation = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    vector = {'translation':'word2vec', 'softcosine':'word2vec'}
    Rerank(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector)