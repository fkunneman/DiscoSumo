__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import ev, metrics
import os

from operator import itemgetter
from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalCosine, SemevalSoftCosine
from semeval_svm import SemevalSVM

from sklearn.metrics import f1_score, accuracy_score

GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'
FEATURE_PATH='feature'
if not os.path.exists(FEATURE_PATH):
    os.mkdir(FEATURE_PATH)
EVALUATION_PATH='evaluation'
if not os.path.exists(EVALUATION_PATH):
    os.mkdir(EVALUATION_PATH)

def prepare_gold(path):
    ir = ev.read_res_file_aid(path, 'trec')
    return ir


def evaluate(ranking):
    gold = ev.read_res_file_aid(GOLD_PATH, 'trec')
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


def run(model, evaluation_path):
    result = model.validate()

    print('Evaluation: ', evaluation_path)
    if len(evaluation_path.split('.')) == 5:
        ranking, y_real, y_pred, parameter_settings = result
        map_baseline, map_model = evaluate(ranking)
        f1score = f1_score(y_real, y_pred)
        accuracy = accuracy_score(y_real, y_pred)
        print('Parameters:', parameter_settings)
        print('MAP baseline: ', map_baseline)
        print('MAP model: ', map_model)
        print('Accuracy: ', accuracy)
        print('F-Score: ', f1score)
        model.save(ranking=ranking, path=path, parameter_settings=parameter_settings)
    else:
        map_baseline, map_model = evaluate(result)
        print('MAP baseline: ', map_baseline)
        print('MAP model: ', map_model)
        print(10 * '-')
        model.save(ranking=result, path=path, parameter_settings='')


if __name__ == '__main__':
    # softcosine / stop / word2vec+elmo
    softcosine = SemevalSoftCosine(stop=True, vector='word2vec+elmo')
    path = os.path.join(EVALUATION_PATH, 'softcosine.stop.word2vec_elmo.ranking')
    run(softcosine, path)
    del softcosine
    ###############################################################################
    # softcosine / nonstop / word2vec+elmo
    softcosine = SemevalSoftCosine(stop=False, vector='word2vec+elmo')
    path = os.path.join(EVALUATION_PATH, 'softcosine.nonstop.word2vec_elmo.ranking')
    run(softcosine, path)
    del softcosine
    ###############################################################################
    # softcosine / stop / fasttext+elmo
    softcosine = SemevalSoftCosine(stop=True, vector='fasttext+elmo')
    path = os.path.join(EVALUATION_PATH, 'softcosine.stop.fasttext_elmo.ranking')
    run(softcosine, path)
    del softcosine
    ###############################################################################
    # softcosine / nonstop / fasttext+elmo
    softcosine = SemevalSoftCosine(stop=False, vector='fasttext+elmo')
    path = os.path.join(EVALUATION_PATH, 'softcosine.nonstop.fasttext_elmo.ranking')
    run(softcosine, path)
    del softcosine
    ###############################################################################
    # regression / softcosine / stop / word2vec+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.word2vec_elmo.features')
    svm = SemevalSVM(model='regression', features='softcosine,', comment_features='softcosine,', stop=True, vector='word2vec+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'regression.softcosine.stop.word2vec_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # regression / softcosine / nonstop / word2vec+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.nonstop.word2vec_elmo.features')
    svm = SemevalSVM(model='regression', features='softcosine,', comment_features='softcosine,', stop=False, vector='word2vec+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'regression.softcosine.nonstop.word2vec_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # svm / softcosine / stop / word2vec+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.word2vec_elmo.features')
    svm = SemevalSVM(model='svm', features='softcosine,', comment_features='softcosine,', stop=True, vector='word2vec+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'svm.softcosine.stop.word2vec_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # svm / softcosine / nonstop / word2vec+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.nonstop.word2vec_elmo.features')
    svm = SemevalSVM(model='svm', features='softcosine,', comment_features='softcosine,', stop=False, vector='word2vec+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'svm.softcosine.nonstop.word2vec_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # regression / softcosine / stop / fasttext+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.fasttext_elmo.features')
    svm = SemevalSVM(model='regression', features='softcosine,', comment_features='softcosine,', stop=True, vector='fasttext+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'regression.softcosine.stop.fasttext_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # regression / softcosine / nonstop / fasttext+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.nonstop.fasttext_elmo.features')
    svm = SemevalSVM(model='regression', features='softcosine,', comment_features='softcosine,', stop=False, vector='fasttext+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'regression.softcosine.nonstop.fasttext_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # svm / softcosine / stop / fasttext+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.fasttext_elmo.features')
    svm = SemevalSVM(model='svm', features='softcosine,', comment_features='softcosine,', stop=True, vector='fasttext+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'svm.softcosine.stop.fasttext_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################
    # svm / softcosine / nonstop / fasttext+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.nonstop.fasttext_elmo.features')
    svm = SemevalSVM(model='svm', features='softcosine,', comment_features='softcosine,', stop=False, vector='fasttext+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'svm.softcosine.nonstop.fasttext_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################