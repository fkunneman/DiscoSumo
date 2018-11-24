__author__='thiagocastroferreira'

import sys
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import copy
import ev, metrics
import os

from operator import itemgetter
from multiprocessing import Pool
from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalCosine, SemevalSoftCosine
from semeval_svm import SemevalSVM

from sklearn.metrics import f1_score, accuracy_score

GOLD_PATH='/roaming/tcastrof/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'
FEATURE_PATH='feature'
if not os.path.exists(FEATURE_PATH):
    os.mkdir(FEATURE_PATH)

EVALUATION_PATH='evaluation'
if not os.path.exists(EVALUATION_PATH):
    os.mkdir(EVALUATION_PATH)

DEV_EVAL_PATH=os.path.join(EVALUATION_PATH, 'dev')
if not os.path.exists(DEV_EVAL_PATH):
    os.mkdir(DEV_EVAL_PATH)

TEST2016_EVAL_PATH=os.path.join(EVALUATION_PATH, 'test2016')
if not os.path.exists(TEST2016_EVAL_PATH):
    os.mkdir(TEST2016_EVAL_PATH)

TEST2017_EVAL_PATH=os.path.join(EVALUATION_PATH, 'test2017')
if not os.path.exists(TEST2017_EVAL_PATH):
    os.mkdir(TEST2017_EVAL_PATH)

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


def run_svm(model_, metric, stop, vector, evaluation_path, feature_path, alpha=0.7, sigma=0.3):
    model = SemevalSVM(model=model_, features=metric, comment_features=metric, stop=stop, vector=vector, path=feature_path, sigma=sigma, alpha=alpha)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.testset2016, model.test2016elmo, model.test2016idx, model.fulltest2016elmo, model.fulltest2016idx)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.testset2017, model.test2017elmo, model.test2017idx, model.fulltest2017elmo, model.fulltest2017idx)
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    ranking, y_real, y_pred, parameter_settings = result_test2016
    model.save(ranking=ranking, path=test2016_path, parameter_settings=parameter_settings)

    ranking, y_real, y_pred, parameter_settings = result_test2017
    model.save(ranking=ranking, path=test2017_path, parameter_settings=parameter_settings)

    ranking, y_real, y_pred, parameter_settings = result_dev
    model.save(ranking=ranking, path=dev_path, parameter_settings=parameter_settings)

    map_baseline, map_model = evaluate(copy.copy(ranking))
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation: ', evaluation_path)
    print('Parameters:', parameter_settings)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)

def run_bm25(stop, evaluation_path):
    model = SemevalBM25(stop=stop)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.testset2016)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.testset2017)
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    model.save(ranking=result_test2016, path=test2016_path, parameter_settings='')

    model.save(ranking=result_test2017, path=test2017_path, parameter_settings='')

    model.save(ranking=result_dev, path=dev_path, parameter_settings='')
    map_baseline, map_model = evaluate(copy.copy(result_dev))

    print('Evaluation: ', evaluation_path)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')
    del model


def run_translation(stop, vector, evaluation_path):
    best = {'map': 0, 'ranking': {}, 'y_real': [], 'y_pred': [], 'parameter_settings': '', 'alpha':0, 'sigma':0, 'model': None}

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for alpha in alphas:
        sigma = abs(1-alpha)
        translation = SemevalTranslation(alpha=alpha, sigma=sigma, vector=vector, stop=stop)
        ranking = translation.validate()

        map_baseline, map_model = evaluate(copy.copy(ranking))
        if map_model > best['map']:
            best['map'] = map_model
            best['ranking'] = ranking
            best['alpha'] = alpha
            best['sigma'] = sigma
            best['parameter_settings'] = 'alpha='+str(alpha)+','+'sigma='+str(sigma)
            best['model'] = translation

    path = os.path.join(DEV_EVAL_PATH, evaluation_path)
    best['model'].save(ranking=best['ranking'], path=path, parameter_settings=best['parameter_settings'])
    print('MAP model: ', best['map'])
    print(10 * '-')

    translation = SemevalTranslation(alpha=best['alpha'], sigma=best['sigma'], vector=vector, stop=stop)
    # test2016
    path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)
    ranking = translation.test(translation.testset2016, translation.test2016elmo, translation.test2016idx, translation.fulltest2016elmo, translation.fulltest2016idx)
    translation.save(ranking=ranking, path=path, parameter_settings=best['parameter_settings'])

    # test2017
    path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)
    ranking = translation.test(translation.testset2017, translation.test2017elmo, translation.test2017idx, translation.fulltest2017elmo, translation.fulltest2017idx)
    translation.save(ranking=ranking, path=path, parameter_settings=best['parameter_settings'])


def run_softcosine(stop, vector, evaluation_path):
    model = SemevalSoftCosine(stop=stop, vector=vector)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.testset2016, model.test2016elmo, model.test2016idx, model.fulltest2016elmo, model.fulltest2016idx)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.testset2017, model.test2017elmo, model.test2017idx, model.fulltest2017elmo, model.fulltest2017idx)
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    model.save(ranking=result_test2016, path=test2016_path, parameter_settings='')

    model.save(ranking=result_test2017, path=test2017_path, parameter_settings='')

    model.save(ranking=result_dev, path=dev_path, parameter_settings='')
    map_baseline, map_model = evaluate(copy.copy(result_dev))

    print('Evaluation: ', evaluation_path)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')


if __name__ == '__main__':
    # THREADS = 4
    # pool = Pool(processes=THREADS)
    #
    #
    # # BM25
    # # stop, evaluation_path
    # # BM25 / stop
    # path = 'bm25.stop.ranking'
    # pool.apply_async(run_bm25, [True, path])
    # ###############################################################################
    # # BM25 / nonstop
    # path = 'bm25.nonstop.ranking'
    # pool.apply_async(run_bm25, [False, path])
    # ###############################################################################
    #
    # # TRANSLATION
    # # stop, vector, evaluation_path
    # # translation / stop / word2vec
    # path = 'translation.stop.word2vec.ranking'
    # pool.apply_async(run_translation, [True, 'word2vec', path])
    # ###############################################################################
    # # translation / stop / fasttext
    # path = 'translation.stop.fasttext.ranking'
    # pool.apply_async(run_translation, [True, 'fasttext', path])
    # ###############################################################################
    # # translation / stop / word2vec+elmo
    # path = 'translation.stop.word2vec_elmo.ranking'
    # pool.apply_async(run_translation, [True, 'word2vec+elmo', path])
    # ###############################################################################
    # # translation / stop / fasttext+elmo
    # path = 'translation.stop.fasttext_elmo.ranking'
    # pool.apply_async(run_translation, [True, 'fasttext+elmo', path])
    # ###############################################################################
    #
    # # SOFTCOSINE
    # # stop, vector, evaluation_path
    # # softcosine / stop / word2vec+elmo
    # path = 'softcosine.stop.word2vec_elmo.ranking'
    # pool.apply_async(run_softcosine, [True, 'word2vec+elmo', path])
    # ###############################################################################
    # # softcosine / nonstop / word2vec+elmo
    # path = 'softcosine.nonstop.word2vec_elmo.ranking'
    # pool.apply_async(run_softcosine, [False, 'word2vec+elmo', path])
    # ###############################################################################
    # # softcosine / stop / fasttext+elmo
    # path = 'softcosine.stop.fasttext_elmo.ranking'
    # pool.apply_async(run_softcosine, [True, 'fasttext+elmo', path])
    # ###############################################################################
    # # softcosine / stop / fasttext
    # path = 'softcosine.stop.fasttext.ranking'
    # pool.apply_async(run_softcosine, [True, 'fasttext', path])
    # ###############################################################################
    # # softcosine / stop / fasttext
    # path = 'softcosine.stop.word2vec.ranking'
    # pool.apply_async(run_softcosine, [True, 'word2vec', path])
    # ###############################################################################
    #
    # # SVM / REGRESSION
    # # model_, metric, stop, vector, evaluation_path, feature_path, alpha=0.7, sigma=0.3
    # # svm / bm25 / stop
    # path = 'svm.bm25.stop.ranking'
    # feature_path = 'bm25.stop.features'
    # pool.apply_async(run_svm, ['svm', 'bm25,', True, '', path, feature_path])
    # ###############################################################################
    # # regression / bm25 / stop
    # path = 'regression.bm25.stop.ranking'
    # feature_path = 'bm25.stop.features'
    # pool.apply_async(run_svm, ['regression', 'bm25,', True, '', path, feature_path])
    # ###############################################################################
    # # svm / softcosine / stop / word2vec+elmo
    # path = 'svm.softcosine.stop.word2vec_elmo.ranking'
    # feature_path = 'softcosine.stop.word2vec_elmo.features'
    # pool.apply_async(run_svm, ['svm', 'softcosine,', True, 'word2vec+elmo', path, feature_path])
    # ###############################################################################
    # # regression / softcosine / stop / word2vec+elmo
    # path = 'regression.softcosine.stop.word2vec_elmo.ranking'
    # feature_path = 'softcosine.stop.word2vec_elmo.features'
    # pool.apply_async(run_svm, ['regression', 'softcosine,', True, 'word2vec+elmo', path, feature_path])
    # ###############################################################################
    # # svm / translation / stop / word2vec
    # path = 'svm.softcosine.stop.ranking'
    # feature_path = 'softcosine.stop.word2vec.features'
    # pool.apply_async(run_svm, ['svm', 'softcosine,', True, 'word2vec', path, feature_path])
    # ###############################################################################