__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import copy
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
        model.save(ranking=ranking, path=path, parameter_settings=parameter_settings)

        map_baseline, map_model = evaluate(copy.copy(ranking))
        f1score = f1_score(y_real, y_pred)
        accuracy = accuracy_score(y_real, y_pred)
        print('Parameters:', parameter_settings)
        print('MAP baseline: ', map_baseline)
        print('MAP model: ', map_model)
        print('Accuracy: ', accuracy)
        print('F-Score: ', f1score)

    else:
        model.save(ranking=result, path=path, parameter_settings='')
        map_baseline, map_model = evaluate(copy.copy(result))

        print('MAP baseline: ', map_baseline)
        print('MAP model: ', map_model)
        print(10 * '-')


def run_translation(model, stop, vector, path, feature_path):
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best = {
        'map': 0,
        'ranking': {},
        'y_real': [],
        'y_pred': [],
        'parameter_settings': '',
        'model': None
    }

    if model == '':
        for alpha in alphas:
            sigma = abs(1-alpha)
            translation = SemevalTranslation(alpha=alpha, sigma=sigma, vector=vector, stop=stop)
            ranking = translation.validate()

            map_baseline, map_model = evaluate(copy.copy(ranking))
            if map_model > best['map']:
                best['map'] = map_model
                best['ranking'] = ranking
                best['parameter_settings'] = 'alpha='+str(alpha)+','+'sigma='+str(sigma)
                best['model'] = translation

        best['model'].save(ranking=best['ranking'], path=path, parameter_settings=best['parameter_settings'])

        print('MAP model: ', best['map'])
        print(10 * '-')
    else:
        for alpha in alphas:
            sigma = abs(1-alpha)
            svm = SemevalSVM(model=model, features='translation,', comment_features='translation,', stop=stop, vector=vector, path=feature_path, alpha=alpha, sigma=sigma)
            ranking, y_real, y_pred, parameter_settings = svm.validate()
            map_baseline, map_model = evaluate(copy.copy(ranking))
            if map_model > best['map']:
                best['map'] = map_model
                best['ranking'] = ranking
                best['y_pred'] = y_pred
                best['y_real'] = y_real
                best['parameter_settings'] = parameter_settings + ',alpha='+str(alpha)+','+'sigma='+str(sigma)
                best['model'] = svm

        best['model'].save(ranking=best['ranking'], path=path, parameter_settings=best['parameter_settings'])
        f1score = f1_score(best['y_real'], best['y_pred'])
        accuracy = accuracy_score(best['y_real'], best['y_pred'])

        print('MAP model: ', best['map'])
        print('Accuracy: ', accuracy)
        print('F-Score: ', f1score)
        print(10 * '-')


if __name__ == '__main__':
    # TRANSLATION
    # translation / stop / word2vec
    path = os.path.join(EVALUATION_PATH, 'translation.stop.word2vec.ranking')
    run_translation(model='', stop=True, vector='word2vec', path=path, feature_path='')
    ###############################################################################
    # translation / stop / fasttext
    path = os.path.join(EVALUATION_PATH, 'translation.stop.fasttext.ranking')
    run_translation(model='', stop=True, vector='fasttext', path=path, feature_path='')
    ###############################################################################
    # translation / stop / word2vec+elmo
    path = os.path.join(EVALUATION_PATH, 'translation.stop.word2vec_elmo.ranking')
    run_translation(model='', stop=True, vector='word2vec+elmo', path=path, feature_path='')
    ###############################################################################
    # translation / stop / fasttext+elmo
    path = os.path.join(EVALUATION_PATH, 'translation.stop.fasttext_elmo.ranking')
    run_translation(model='', stop=True, vector='fasttext+elmo', path=path, feature_path='')
    ###############################################################################

    # BM25
    # BM25 / stop
    softcosine = SemevalBM25(stop=True)
    path = os.path.join(EVALUATION_PATH, 'bm25.stop.ranking')
    run(softcosine, path)
    del softcosine
    ###############################################################################
    # BM25 / nonstop
    softcosine = SemevalBM25(stop=True)
    path = os.path.join(EVALUATION_PATH, 'bm25.nonstop.ranking')
    run(softcosine, path)
    del softcosine
    ###############################################################################

    # SOFTCOSINE
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

    # COMMENTS
    # BM25
    # svm / bm25 / stop
    path = os.path.join(EVALUATION_PATH, 'svm.bm25.stop.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'bm25.stop.ranking.features')
    svm = SemevalSVM(model='svm', features='bm25,', comment_features='bm25,', stop=True, vector='word2vec', path=feature_path)
    run(svm, path)
    del svm
    ###############################################################################
    # regression / bm25 / stop
    path = os.path.join(EVALUATION_PATH, 'regression.bm25.stop.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'bm25.stop.ranking.features')
    regression = SemevalSVM(model='regression', features='bm25,', comment_features='bm25,', stop=True, vector='word2vec', path=feature_path)
    run(regression, path)
    ###############################################################################

    # SOFTCOSINE
    # regression / softcosine / stop / word2vec+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.word2vec_elmo.features')
    svm = SemevalSVM(model='regression', features='softcosine,', comment_features='softcosine,', stop=True, vector='word2vec+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'regression.softcosine.stop.word2vec_elmo.ranking')
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
    # svm / softcosine / stop / fasttext+elmo
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.stop.fasttext_elmo.features')
    svm = SemevalSVM(model='svm', features='softcosine,', comment_features='softcosine,', stop=True, vector='fasttext+elmo', path=feature_path)
    path = os.path.join(EVALUATION_PATH, 'svm.softcosine.stop.fasttext_elmo.ranking')
    run(svm, path)
    del svm
    ###############################################################################

    # TRANSLATION
    # regression / translation / stop / word2vec
    path = os.path.join(EVALUATION_PATH, 'regression.translation.stop.word2vec.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.word2vec.features')
    run_translation(model='regression', stop=True, vector='word2vec', path=path, feature_path=feature_path)
    ###############################################################################
    # svm / translation / stop / word2vec
    path = os.path.join(EVALUATION_PATH, 'svm.translation.stop.word2vec.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.word2vec.features')
    run_translation(model='svm', stop=True, vector='word2vec', path=path, feature_path=feature_path)
    ##############################################################################
    # regression / translation / stop / fasttext
    path = os.path.join(EVALUATION_PATH, 'regression.translation.stop.fasttext.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.fasttext.features')
    run_translation(model='regression', stop=True, vector='fasttext', path=path, feature_path=feature_path)
    ###############################################################################
    # svm / translation / stop / fasttext
    path = os.path.join(EVALUATION_PATH, 'svm.translation.stop.fasttext.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.fasttext.features')
    run_translation(model='svm', stop=True, vector='fasttext', path=path, feature_path=feature_path)
    ###############################################################################
    # regression / translation / stop / word2vec+elmo
    path = os.path.join(EVALUATION_PATH, 'regression.translation.stop.word2vec_elmo.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.word2vec_elmo.features')
    run_translation(model='regression', stop=True, vector='word2vec+elmo', path=path, feature_path=feature_path)
    ###############################################################################
    # svm / translation / stop / word2vec+elmo
    path = os.path.join(EVALUATION_PATH, 'svm.translation.stop.word2vec_elmo.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.word2vec_elmo.features')
    run_translation(model='svm', stop=True, vector='word2vec+elmo', path=path, feature_path=feature_path)
    ###############################################################################
    # regression / translation / stop / fasttext+elmo
    path = os.path.join(EVALUATION_PATH, 'regression.translation.stop.fasttext_elmo.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.fasttext_elmo.features')
    run_translation(model='regression', stop=True, vector='fasttext+elmo', path=path, feature_path=feature_path)
    ###############################################################################
    # svm / translation / stop / fasttext+elmo
    path = os.path.join(EVALUATION_PATH, 'svm.translation.stop.fasttext_elmo.ranking')
    feature_path = os.path.join(FEATURE_PATH, 'translation.stop.fasttext_elmo.features')
    run_translation(model='svm', stop=True, vector='fasttext+elmo', path=path, feature_path=feature_path)
    ###############################################################################