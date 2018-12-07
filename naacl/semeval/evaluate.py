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
from semeval_ensemble import SemevalEnsemble

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

    result_test2016 = model.test(model.test2016data, model.test2016idx, model.test2016elmo, test_='test2016')
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.test2017data, model.test2017idx, model.test2017elmo, test_='test2017')
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


def run_bm25(stop, lowercase, punctuation, proctrain, evaluation_path):
    model = SemevalBM25(stop=stop, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.test2016data)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.test2017data)
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


def run_translation(stop, lowercase, punctuation, proctrain, vector, evaluation_path, alpha=0.0, sigma=0.0):
    best = { 'map': 0.0 }
    translation = SemevalTranslation(alpha=alpha, sigma=sigma, punctuation=punctuation, proctrain=proctrain, vector=vector, stop=stop, lowercase=lowercase)

    if alpha > 0 and sigma > 0:
        translation.set_parameters(alpha=alpha, sigma=sigma)
        ranking = translation.validate()

        map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))
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
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    ranking = translation.test(translation.test2016data, translation.test2016idx, translation.test2016elmo)
    translation.save(ranking=ranking, path=path, parameter_settings=best['parameter_settings'])

    # test2017
    path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)
    ranking = translation.test(translation.test2017data, translation.test2017idx, translation.test2017elmo)
    translation.save(ranking=ranking, path=path, parameter_settings=best['parameter_settings'])
    return best['alpha'], best['sigma']


def run_softcosine(stop, lowercase, punctuation, proctrain, vector, evaluation_path):
    model = SemevalSoftCosine(stop=stop, vector=vector, lowercase=lowercase, punctuation=punctuation, proctrain=proctrain)
    result_dev = model.validate()
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(model.test2016data, model.test2016idx, model.test2016elmo)
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(model.test2017data, model.test2017idx, model.test2017elmo)
    test2017_path = os.path.join(TEST2017_EVAL_PATH, evaluation_path)

    model.save(ranking=result_test2016, path=test2016_path, parameter_settings='')

    model.save(ranking=result_test2017, path=test2017_path, parameter_settings='')

    model.save(ranking=result_dev, path=dev_path, parameter_settings='')
    map_baseline, map_model = evaluate(copy.copy(result_dev), prepare_gold(DEV_GOLD_PATH))

    print('Evaluation: ', evaluation_path)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')


def run_ensemble(stop, lowercase, punctuation, vector, evaluation_path, kernel_path, alpha, sigma):
    model = SemevalEnsemble(stop=stop, lowercase=lowercase, punctuation=punctuation, vector=vector, scale=True, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    result_dev = model.test(set_='dev')
    dev_path = os.path.join(DEV_EVAL_PATH, evaluation_path)

    result_test2016 = model.test(set_='test2016')
    test2016_path = os.path.join(TEST2016_EVAL_PATH, evaluation_path)

    result_test2017 = model.test(set_='test2017')
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


if __name__ == '__main__':
    # PREPROCESSING EXPERIMENTS
    ###############################################################################
    # ENSEMBLE
    # lowercase, stop, punctuation
    vector = {'translation':'alignments', 'softcosine':'word2vec', 'kernel':'word2vec'}
    lower = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    stop = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    punctuation = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    kernel_path = 'kernel.word2vec.lower.pickle'
    path = 'ensemble.lower.stop.punct.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase, stop
    lower = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    stop = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    punctuation = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    kernel_path = 'kernel.word2vec.lower.pickle'
    path = 'ensemble.lower.stop.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase, punctuation,
    lower = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    stop = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    punctuation = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    kernel_path = 'kernel.word2vec.lower.pickle'
    path = 'ensemble.lower.punct.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # stop, punctuation
    lower = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    stop = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    punctuation = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    kernel_path = 'kernel.word2vec.pickle'
    path = 'ensemble.stop.punct.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase
    lower = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    stop = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    punctuation = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    kernel_path = 'kernel.word2vec.lower.pickle'
    path = 'ensemble.lower.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # stop
    lower = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    stop = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    punctuation = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    kernel_path = 'kernel.word2vec.pickle'
    path = 'ensemble.stop.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # punctuation
    lower = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    stop = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    punctuation = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    kernel_path = 'kernel.word2vec.pickle'
    path = 'ensemble.punct.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    #
    lower = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    stop = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    punctuation = {'bm25':False, 'translation':False, 'softcosine':False, 'kernel':False}
    kernel_path = 'kernel.word2vec.pickle'
    path = 'ensemble.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    ###############################################################################


    # WORD SIMILARITY EXPERIMENTS
    ###############################################################################
    # ENSEMBLE
    lower = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':False}
    stop = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    punctuation = {'bm25':True, 'translation':True, 'softcosine':True, 'kernel':True}
    # lowercase, stop, punctuation, alignments
    vector = {'translation':'alignments', 'softcosine':'alignments', 'kernel': 'alignments'}
    kernel_path = 'kernel.alignments.pickle'
    path = 'ensemble.lower.stop.punct.alignments.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase, stop, punctuation, word2vec
    vector = {'translation':'word2vec', 'softcosine':'word2vec', 'kernel': 'word2vec'}
    kernel_path = 'kernel.word2vec.pickle'
    path = 'ensemble.lower.stop.punct.word2vec.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase, stop, punctuation, fasttext
    vector = {'translation':'fasttext', 'softcosine':'fasttext', 'kernel': 'fasttext'}
    kernel_path = 'kernel.fasttext.pickle'
    path = 'ensemble.lower.stop.punct.fasttext.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase, stop, punctuation, word2vec+elmo
    vector = {'translation':'word2vec+elmo', 'softcosine':'word2vec+elmo', 'kernel': 'word2vec+elmo'}
    kernel_path = 'kernel.word2vec+elmo.pickle'
    path = 'ensemble.lower.stop.punct.word2vec+elmo.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)
    # lowercase, stop, punctuation, fasttext+elmo
    vector = {'translation':'fasttext+elmo', 'softcosine':'fasttext+elmo', 'kernel': 'fasttext+elmo'}
    kernel_path = 'kernel.fasttext+elmo.pickle'
    path = 'ensemble.lower.stop.punct.fasttext+elmo.ranking'
    alpha, sigma = run_translation(lowercase=lower['translation'], stop=stop['translation'], punctuation=punctuation['translation'], proctrain=True, vector=vector['translation'], evaluation_path='translation.tmp')
    run_ensemble(stop=stop, lowercase=lower, punctuation=punctuation, vector=vector, evaluation_path=path, kernel_path=kernel_path, alpha=alpha, sigma=sigma)