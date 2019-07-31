__author__='thiagocastroferreira'

import _pickle as p
import json
import os
import time

from main import GoeieVraag
from gensim.corpora import Dictionary
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

DATA_PATH='/roaming/fkunnema/goeievraag/data/'
TRAINING_DATA='/roaming/fkunnema/goeievraag/data/ranked_questions_labeled_proc.json'
CORPUS_PATH=os.path.join(DATA_PATH, 'corpus.json')
NEW_ANSWERS=os.path.join(DATA_PATH, 'answer_parsed_proc.json')

def mean_avg_prec(ranking, n=10):
    map_ = 0.0
    for qid in ranking:
        result = sorted(ranking[qid], key=lambda x: x[1], reverse=True)[:n]

        precision = []
        for i, row in enumerate(result):
            size = i+1
            tp = len([w for w in result[:i+1] if w[2] == 1])

            precision.append(float(tp) / size)

        map_ += sum(precision) / len(precision)

    return map_ / 20


def eval_retrieval(goeie):
    testdata = goeie.testdata

    # BM25
    bm25time = []
    bm25ranking = {}

    results = []
    allacc, acc10, acc30 = [], [], []
    for q1id in testdata:
        bm25ranking[q1id] = []

        auxid = list(testdata[q1id].keys())[0]
        q1 = testdata[q1id][auxid]['q1']

        # categories = [c[1] for c in goeie.question2cat(' '.join(q1)) if c[1] != '15']
        # print(' '.join(q1).encode('utf-8'),categories)
        start = time.time()
        questions = goeie.retrieve(q1)
        end = time.time()
        bm25time.append(end-start)

        questions = sorted(questions, key=lambda x: x['score'])
        qids = [q['id'] for q in questions]

        qtokens = [(q['id'], ' '.join(q['tokens'])) for q in questions]
        for q in qtokens:
            results.append(','.join([str(q1id), str(q[0]), q[1]]))

        all = [qid for qid in testdata[q1id] if qid in qids]
        acc30_ = [qid for qid in testdata[q1id] if qid in qids and testdata[q1id][qid]['label'] == 1]
        acc10_ = [qid for qid in testdata[q1id] if qid in qids[:10] and testdata[q1id][qid]['label'] == 1]

        allacc.append((len(all), 10))
        acc30.append((len(acc30_), 10))
        acc10.append((len(acc10_), 10))

    num, dem = sum([w[0] for w in allacc]), float(sum([w[1] for w in allacc]))
    print('All accuracy:', round(num / dem, 5))
    num, dem = sum([w[0] for w in acc10]), float(sum([w[1] for w in allacc]))
    print('10 accuracy:', round(num / dem, 5))
    num, dem = sum([w[0] for w in acc30]), float(sum([w[1] for w in allacc]))
    print('30 accuracy:', round(num / dem, 5))

    with open('retrieve_results.txt', 'w') as f:
        f.write('\n'.join(results))


def eval_reranking(goeie):
    print('Reranking...')
    procdata = goeie.procdata
    answers = json.load(open(NEW_ANSWERS))

    folds = []

    procids = list(procdata.keys())
    kf = KFold(n_splits=5)
    for trainids, testids in kf.split(procids):
        traindata = {}
        for i in trainids:
            qid = procids[i]
            traindata[qid] = procdata[qid]
        goeie.traindata = traindata

        testdata = {}
        for i in testids:
            qid = procids[i]
            testdata[qid] = procdata[qid]
        goeie.testdata = testdata

        corpus = []
        for qid in goeie.questions:
            if qid not in goeie.testdata:
                question = goeie.seeds[goeie.seed2idx[qid]]
                corpus.append(question['tokens'])
        for answer in answers.values():
            corpus.append(answer['tokens_proc'])
        goeie.dict = Dictionary(corpus)  # fit dictionary

        goeie.init_bm25(goeie.seeds)
        goeie.load_sofcos()

        best, best_map = {'alpha':0.0, 'sigma':0.0}, -1
        transranking = {}
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            sigma = 1 - alpha
            goeie.init_translation(corpus=corpus, alpha=alpha, sigma=sigma)

            for q1id in list(traindata.keys())[:20]:
                transranking[q1id] = []

                for q2id in traindata[q1id]:
                    score, label = traindata[q1id][q2id]['score'], traindata[q1id][q2id]['label']

                    q1, q2 = traindata[q1id][q2id]['q1'], traindata[q1id][q2id]['q2']
                    q1, q1emb, q2, q2emb = goeie.preprocess(q1=q1, q2=q2)
                    transcore = goeie.translate(q1, q1emb, q2, q2emb)
                    transranking[q1id].append((q2id, transcore, label))

            map_ = mean_avg_prec(transranking)
            print('Translation - alpha: {0} \t sigma: {1} \t MAP: {2}'.format(alpha, sigma, map_))
            if map_ > best_map:
                best_map = map_
                best = {'alpha':alpha, 'sigma':sigma}

        goeie.init_translation(corpus=corpus, alpha=best['alpha'], sigma=best['sigma'])
        goeie.init_ensemble()

        bm25time, transtime, softtime, enstime = [], [], [], []
        ranking = {}
        bm25ranking, transranking, softranking, ensranking = {}, {}, {}, {}
        for q1id in testdata:
            bm25ranking[q1id] = []
            transranking[q1id] = []
            softranking[q1id] = []
            ensranking[q1id] = []
            ranking[q1id] = []

            for q2id in testdata[q1id]:
                score, label = testdata[q1id][q2id]['score'], testdata[q1id][q2id]['label']
                bm25ranking[q1id].append((q2id, score, label))

                q1, q2 = testdata[q1id][q2id]['q1'], testdata[q1id][q2id]['q2']
                q1, q1emb, q2, q2emb = goeie.preprocess(q1=q1, q2=q2)

                start = time.time()
                transcore = goeie.translate(q1, q1emb, q2, q2emb)
                end = time.time()
                transtime.append(end-start)
                transranking[q1id].append((q2id, transcore, label))

                start = time.time()
                softscore = goeie.softcos(q1, q1emb, q2, q2emb)
                end = time.time()
                softtime.append(end-start)
                softranking[q1id].append((q2id, softscore, label))

                start = time.time()
                enscore = goeie.ensembling(q1, q1emb, q2id, q2, q2emb)
                end = time.time()
                enstime.append(end-start)
                ensranking[q1id].append((q2id, enscore, label))

                ranking[q1id].append((q2id, label, label))

        folds.append({
            'upper': ranking,
            'bm25': bm25ranking,
            'translation': transranking,
            'softcosine': softranking,
            'ensemble': ensranking
        })

        print('Upper bound: Evaluation')
        print('MAP:', round(mean_avg_prec(ranking), 4))
        print(10 * '-')

        print('BM25: Evaluation')
        print('MAP:', round(mean_avg_prec(bm25ranking), 4))
        print(10 * '-')

        print('Translation: Evaluation')
        print('MAP:', round(mean_avg_prec(transranking), 4))
        print('Time: ', round(sum(transtime) / len(transtime), 4))
        print(10 * '-')

        print('Softcosine: Evaluation')
        print('MAP:', round(mean_avg_prec(softranking), 4))
        print('Time: ', round(sum(softtime) / len(softtime), 4))
        print(10 * '-')

        print('Ensembling: Evaluation')
        print('MAP:', round(mean_avg_prec(ensranking), 4))
        print('Time: ', round(sum(enstime) / len(enstime), 4))
        print(10 * '-')
        print(50 * '*')
    p.dump(folds, open('folds.pickle', 'wb'))


if __name__ == '__main__':
    goeie = GoeieVraag(evaluation=True)

    # eval_retrieval(goeie)
    #
    # print(10 * '*')

    eval_reranking(goeie)

