__author__='thiagocastroferreira'

import sys
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import copy
import ev, metrics
import os

from operator import itemgetter
from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalCosine, SemevalSoftCosine
from semeval_svm import SemevalSVM
from semeval_kernel import SemevalTreeKernel

from sklearn.metrics import f1_score, accuracy_score

DEV_GOLD_PATH='/roaming/tcastrof/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'
TEST2016_GOLD_PATH='/roaming/tcastrof/semeval/evaluation/SemEval2016-Task3-CQA-QL-test2016.xml.subtaskB.relevancy'
TEST2017_GOLD_PATH='/roaming/tcastrof/semeval/evaluation/SemEval2016-Task3-CQA-QL-test2017.xml.subtaskB.relevancy'

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


def run_kernel(smoothed, vector, tree, lowercase, evaluation_path, kernel_path):
    model = SemevalTreeKernel(smoothed=smoothed, vector=vector, tree=tree, kernel_path=kernel_path, lowercase=lowercase)

    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.testset2016, model.fulltest2016idx, model.fulltest2016elmo, test_='test2016')
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.testset2017, model.fulltest2017idx, model.fulltest2017elmo, test_='test2017')
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    ranking, y_real, y_pred, parameter_settings = result_test2016
    model.save(ranking=ranking, path=test2016_path, parameter_settings=parameter_settings)

    ranking, y_real, y_pred, parameter_settings = result_test2017
    model.save(ranking=ranking, path=test2017_path, parameter_settings=parameter_settings)

    ranking, y_real, y_pred, parameter_settings = result_dev
    model.save(ranking=ranking, path=dev_path, parameter_settings=parameter_settings)

    map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation: ', evaluation_path)
    print('Parameters:', parameter_settings)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)


def run_svm(model_, metric, stop, vector, lowercase, evaluation_path, feature_path, alpha=0.7, sigma=0.3, gridsearch='brutal'):
    model = SemevalSVM(model=model_, features=metric, comment_features=metric, stop=stop, vector=vector, lowercase=lowercase, path=feature_path, sigma=sigma, alpha=alpha, gridsearch=gridsearch)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.test2016data, model.test2016idx, model.test2016elmo, model.fulltest2016idx, model.fulltest2016elmo, test_='test2016')
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.test2017data, model.test2017idx, model.test2017elmo, model.fulltest2017idx, model.fulltest2017elmo, test_='test2017')
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    ranking, y_real, y_pred, parameter_settings = result_test2016
    model.save(ranking=ranking, path=test2016_path, parameter_settings=parameter_settings)

    ranking, y_real, y_pred, parameter_settings = result_test2017
    model.save(ranking=ranking, path=test2017_path, parameter_settings=parameter_settings)

    ranking, y_real, y_pred, parameter_settings = result_dev
    model.save(ranking=ranking, path=dev_path, parameter_settings=parameter_settings)

    map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation: ', evaluation_path)
    print('Parameters:', parameter_settings)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)


def run_bm25(stop, lowercase, evaluation_path):
    model = SemevalBM25(stop=stop, lowercase=lowercase)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.testset2016)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.testset2017)
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    model.save(ranking=result_test2016, path=test2016_path, parameter_settings='')

    model.save(ranking=result_test2017, path=test2017_path, parameter_settings='')

    model.save(ranking=result_dev, path=dev_path, parameter_settings='')
    map_baseline, map_model = evaluate(copy.copy(result_dev), prepare_gold(DEV_GOLD_PATH))

    print('Evaluation: ', evaluation_path)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')
    del model


def run_translation(stop, lowercase, vector, evaluation_path):
    best = {'map': 0, 'ranking': {}, 'y_real': [], 'y_pred': [], 'parameter_settings': '', 'alpha':0, 'sigma':0, 'model': None}

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector=vector, stop=stop, lowercase=lowercase)
    for alpha in alphas:
        sigma = abs(1-alpha)
        translation.set_parameters(alpha=alpha, sigma=sigma)
        ranking = translation.validate()

        map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))
        if map_model > best['map']:
            best['map'] = copy.copy(map_model)
            best['ranking'] = ranking
            best['alpha'] = alpha
            best['sigma'] = sigma
            best['parameter_settings'] = 'alpha='+str(translation.alpha)+','+'sigma='+str(translation.sigma)
            best['model'] = translation
            print('Parameters: ', best['parameter_settings'])
            print('MAP model: ', best['map'])
            print(10 * '-')
        else:
            print('Not best:')
            print('Parameters: ', 'alpha='+str(translation.alpha)+','+'sigma='+str(translation.sigma))
            print('MAP model: ', map_model)
            print(10 * '-')

    path = os.path.join(DEV_EVAL_PATH, evaluation_path)
    best['model'].save(ranking=best['ranking'], path=path, parameter_settings=best['parameter_settings'])
    print('MAP model: ', best['map'])
    print(10 * '-')

    translation.alpha = best['alpha']
    translation.sigma = best['sigma']
    # test2016
    path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)
    ranking = translation.test(translation.testset2016, translation.test2016idx, translation.test2016elmo, translation.fulltest2016idx, translation.fulltest2016elmo)
    translation.save(ranking=ranking, path=path, parameter_settings=best['parameter_settings'])

    # test2017
    path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)
    ranking = translation.test(translation.testset2017, translation.test2017idx, translation.test2017elmo, translation.fulltest2017idx, translation.fulltest2017elmo)
    translation.save(ranking=ranking, path=path, parameter_settings=best['parameter_settings'])


def run_softcosine(stop, lowercase, vector, evaluation_path):
    model = SemevalSoftCosine(stop=stop, vector=vector, lowercase=lowercase)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.testset2016, model.test2016idx, model.test2016elmo, model.fulltest2016idx, model.fulltest2016elmo)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.testset2017, model.test2017idx, model.test2017elmo, model.fulltest2017idx, model.fulltest2017elmo)
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    model.save(ranking=result_test2016, path=test2016_path, parameter_settings='')

    model.save(ranking=result_test2017, path=test2017_path, parameter_settings='')

    model.save(ranking=result_dev, path=dev_path, parameter_settings='')
    map_baseline, map_model = evaluate(copy.copy(result_dev), prepare_gold(DEV_GOLD_PATH))

    print('Evaluation: ', evaluation_path)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')


if __name__ == '__main__':
    # TRANSLATION
    # stop, vector, evaluation_path
    # translation / stop / word2vec+elmo
    path = 'translation.stop.alignments.ranking'
    run_translation(stop=True, lowercase=True, vector='alignments', evaluation_path=path)
    ###############################################################################
    # translation / stop / word2vec+elmo
    path = 'translation.stop.word2vec_elmo.ranking'
    run_translation(stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path)
    ###############################################################################
    # translation / stop / fasttext+elmo
    path = 'translation.stop.fasttext_elmo.ranking'
    run_translation(stop=True, lowercase=True, vector='fasttext+elmo', evaluation_path=path)
    ###############################################################################
    # translation / stop / word2vec
    path = 'translation.stop.word2vec.ranking'
    run_translation(stop=True, lowercase=True, vector='word2vec', evaluation_path=path)
    ###############################################################################
    # translation / stop / fasttext
    path = 'translation.stop.fasttext.ranking'
    run_translation(stop=True, lowercase=True, vector='fasttext', evaluation_path=path)
    ###############################################################################

    # BM25
    # stop, evaluation_path
    # BM25 / stop
    path = 'bm25.stop.ranking'
    run_bm25(stop=True, lowercase=True, evaluation_path=path)
    ###############################################################################
    # BM25 / nonstop
    path = 'bm25.nonstop.ranking'
    run_bm25(stop=False, lowercase=True, evaluation_path=path)
    ###############################################################################

    # SOFTCOSINE
    # stop, vector, evaluation_path
    # softcosine / stop / word2vec+elmo
    path = 'softcosine.stop.word2vec_elmo.ranking'
    run_softcosine(stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path)
    ###############################################################################
    # softcosine / nonstop / word2vec+elmo
    path = 'softcosine.nonstop.word2vec_elmo.ranking'
    run_softcosine(stop=False, lowercase=True, vector='word2vec+elmo', evaluation_path=path)
    ###############################################################################
    # softcosine / stop / fasttext+elmo
    path = 'softcosine.stop.fasttext_elmo.ranking'
    run_softcosine(stop=True, lowercase=True, vector='fasttext+elmo', evaluation_path=path)
    ###############################################################################
    # softcosine / stop / fasttext
    path = 'softcosine.stop.fasttext.ranking'
    run_softcosine(stop=True, lowercase=True, vector='fasttext', evaluation_path=path)
    ###############################################################################
    # softcosine / stop / fasttext
    path = 'softcosine.stop.word2vec.ranking'
    run_softcosine(stop=True, lowercase=True, vector='word2vec', evaluation_path=path)
    ###############################################################################

    # SVM / REGRESSION
    # model_, metric, stop, vector, evaluation_path, feature_path, alpha=0.9, sigma=0.1
    # svm / translation / stop / word2vec
    path = 'svm.translation.stop.word2vec.ranking'
    feature_path = 'translation.stop.word2vec.features'
    run_svm(model_='svm', metric='translation,', stop=True, lowercase=True, vector='word2vec', evaluation_path=path, feature_path=feature_path, sigma=0.3, alpha=0.7)
    ###############################################################################
    # regression / translation / stop / word2vec
    path = 'regression.translation.stop.word2vec.ranking'
    feature_path = 'translation.stop.word2vec.features'
    run_svm(model_='regression', metric='translation,', stop=True, lowercase=True, vector='word2vec', evaluation_path=path, feature_path=feature_path, sigma=0.3, alpha=0.7)
    ###############################################################################
    # svm / translation / stop / word2vec+elmo
    path = 'svm.translation.stop.word2vec_elmo.ranking'
    feature_path = 'translation.stop.word2vec_elmo.features'
    run_svm(model_='svm', metric='translation,', stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path, sigma=0.1, alpha=0.9)
    ###############################################################################
    # regression / translation / stop / word2vec+elmo
    path = 'regression.translation.stop.word2vec_elmo.ranking'
    feature_path = 'translation.stop.word2vec_elmo.features'
    run_svm(model_='regression', metric='translation,', stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path, sigma=0.1, alpha=0.9)
    ###############################################################################
    # svm / bm25 / stop
    path = 'svm.bm25.stop.ranking'
    feature_path = 'bm25.stop.features'
    run_svm(model_='svm', metric='bm25,', stop=True, lowercase=True, vector='', evaluation_path=path, feature_path=feature_path)
    ###############################################################################
    # regression / bm25 / stop
    path = 'regression.bm25.stop.ranking'
    feature_path = 'bm25.stop.features'
    run_svm(model_='regression', metric='bm25,', stop=True, lowercase=True, vector='', evaluation_path=path, feature_path=feature_path)
    ###############################################################################
    # svm / softcosine / stop / word2vec+elmo
    path = 'svm.softcosine.stop.word2vec_elmo.ranking'
    feature_path = 'softcosine.stop.word2vec_elmo.features'
    run_svm(model_='svm', metric='softcosine,', stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path)
    ###############################################################################
    # regression / softcosine / stop / word2vec+elmo
    path = 'regression.softcosine.stop.word2vec_elmo.ranking'
    feature_path = 'softcosine.stop.word2vec_elmo.features'
    run_svm(model_='regression', metric='softcosine,', stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path)
    # ###############################################################################
    # svm / softcosine / stop / word2vec
    path = 'svm.softcosine.stop.word2vec.ranking'
    feature_path = 'softcosine.stop.word2vec.features'
    run_svm(model_='svm', metric='softcosine,', stop=True, lowercase=True, vector='word2vec', evaluation_path=path, feature_path=feature_path)
    ###############################################################################
    # regression / softcosine / stop / word2vec
    path = 'regression.softcosine.stop.word2vec.ranking'
    feature_path = 'softcosine.stop.word2vec.features'
    run_svm(model_='regression', metric='softcosine,', stop=True, lowercase=True, vector='word2vec', evaluation_path=path, feature_path=feature_path)
    ###############################################################################
    # svm / all / stop / word2vec+elmo
    path = 'svm.all.stop.word2vec_elmo.ranking'
    feature_path = 'all.stop.word2vec_elmo.features'
    run_svm(model_='svm', metric='softcosine,bm25,translation', stop=True, lowercase=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path)
    ###############################################################################
    # regression / all / stop / word2vec+elmo
    path = 'regression.all.stop.word2vec_elmo.ranking'
    feature_path = 'all.stop.word2vec_elmo.features'
    run_svm(model_='regression', metric='softcosine,bm25,translation,', stop=True, lowercase=True, vector='word2vec+elmo',evaluation_path= path, feature_path=feature_path)
    # ###############################################################################

    # TREE KERNEL
    # # svm kernel / subj tree / non smoothed
    # path = 'kernel.subj_tree.ranking'
    # kernel_path = 'kernel.pickle'
    # run_kernel(kernel_path=kernel_path, evaluation_path=path, smoothed=False, vector='', tree='subj_tree')
    # # svm kernel / subj tree / word2vec
    # path = 'kernel.subj_tree.word2vec.ranking'
    # kernel_path = 'kernel.word2vec.pickle'
    # run_kernel(kernel_path=kernel_path, evaluation_path=path, smoothed=True, vector='word2vec', tree='subj_tree')
    # svm kernel / subj tree / word2vec+elmo
    # path = 'kernel.subj_tree.word2vec_elmo.ranking'
    # kernel_path = 'kernel.word2vec+elmo.pickle'
    # run_kernel(kernel_path=kernel_path, evaluation_path=path, smoothed=True, vector='word2vec+elmo', tree='subj_tree')