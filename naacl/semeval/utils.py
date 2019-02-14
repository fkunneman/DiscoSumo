__author__='thiagocastroferreira'

import sys
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import ev, metrics
import paths

DEV_GOLD_PATH=paths.DEV_GOLD_PATH
TEST2016_GOLD_PATH=paths.TEST2016_GOLD_PATH
TEST2017_GOLD_PATH=paths.TEST2017_GOLD_PATH

from operator import itemgetter

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