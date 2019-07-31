__author__='thiagocastroferreira'

import paths
import sys
sys.path.append(paths.MAP_scripts)
import copy
import ev, metrics
from operator import itemgetter
import os

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

def run(path, set_, winner=False):
    with open(path) as f:
        doc = f.read().split('\n')
        print(doc[0])
        doc = [w.strip().split('\t') for w in doc[1:-1]]

    ranking = {}
    for i, row in enumerate(doc):
        q1id, q2id, score, label = row[0], row[1], float(row[3]), row[4]
        #         label = 1 if label == 'true' else 0

        if q1id not in ranking:
            ranking[q1id] = []
        ranking[q1id].append((label, score, q2id))

    if set_ == 'dev':
        set_path = DEV_GOLD_PATH
    elif set_ == 'test2016':
        set_path = TEST2016_GOLD_PATH
    else:
        set_path = TEST2017_GOLD_PATH
    gold = prepare_gold(set_path)

    y_real, y_pred = [], []
    for qid in gold:
        for row in gold[qid]:
            real = 1 if row[0] == 'true' else 0
            y_real.append(real)

            if winner:
                pred = 1 if [w[0] for w in ranking[qid] if w[2] == row[2]][0] == 'true' else 0
            else:
                pred = 1 if float([w[1] for w in ranking[qid] if w[2] == row[2]][0]) > 0 else 0
            y_pred.append(pred)

    map_baseline, map_model = evaluate(copy.copy(ranking), gold)
    return map_model, f1_score(y_real, y_pred), accuracy_score(y_real, y_pred), y_real, y_pred

if __name__ == '__main__':
    print('TEST 2016')
    set_='test2016'

    result_path = 'results'
    for model in os.listdir(result_path):
        path = os.path.join(result_path, model, set_)
        for fname in os.listdir(path):
            path_ = os.path.join(path, fname)
            map_, fscore, accuracy, y_real, y_pred = run(path_, set_, True)
            print(path_.split('/')[-1], 'MAP: ', round(map_, 4), 'F-Score: ', round(fscore, 4), 'Acc.:', round(accuracy, 4))
            print(10 * '-')

    print('\n\n')
    print('TEST 2017')
    set_='test2017'
    result_path = 'results'
    for model in os.listdir(result_path):
        path = os.path.join(result_path, model, set_)
        for fname in os.listdir(path):
            path_ = os.path.join(path, fname)
            map_, fscore, accuracy, y_real, y_pred = run(path_, set_, True)
            print(path_.split('/')[-1], 'MAP: ', round(map_, 4), 'F-Score: ', round(fscore, 4), 'Acc.:', round(accuracy, 4))
            print(10 * '-')