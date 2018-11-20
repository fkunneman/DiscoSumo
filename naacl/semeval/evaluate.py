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

if __name__ == '__main__':
    ###############################################################################
    # bm25
    feature_path = os.path.join(FEATURE_PATH, 'bm25.features')
    svm = SemevalSVM(features='bm25,', comment_features='bm25,', vector='word2vec', path=feature_path)
    ranking, y_real, y_pred, parameter_settings = svm.validate()

    path = os.path.join(EVALUATION_PATH, 'bm25.comments.ranking')
    svm.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation BM25 + Comments')
    print('Parameters:', parameter_settings)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)
    print(10 * '-')

    del svm
    ###############################################################################
    # word2vec translation
    feature_path = os.path.join(FEATURE_PATH, 'translation.word2vec.features')
    svm = SemevalSVM(features='translation,', comment_features='translation,', vector='word2vec', path=feature_path)
    ranking, y_real, y_pred, parameter_settings = svm.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.word2vec.comments.ranking')
    svm.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation Translation Word2Vec + Comments')
    print('Parameters:', parameter_settings)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)
    print(10 * '-')

    del svm
    ###############################################################################
    # word2vec softcosine
    feature_path = os.path.join(FEATURE_PATH, 'softcosine.word2vec_elmo.features')
    svm = SemevalSVM(features='softcosine,', comment_features='softcosine,', vector='word2vec+elmo', path=feature_path)
    ranking, y_real, y_pred, parameter_settings = svm.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.word2vec_elmo.comments.ranking')
    svm.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation Translation Word2Vec + Comments')
    print('Parameters:', parameter_settings)
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)
    print(10 * '-')

    del svm
    ###############################################################################
    # bm25
    bm25 = SemevalBM25()
    ranking = bm25.validate()

    path = os.path.join(EVALUATION_PATH, 'bm25.ranking')
    bm25.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation BM25')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del bm25
    ###############################################################################
    # align translation
    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector='alignments')
    ranking = translation.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.align.ranking')
    translation.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Align Translation')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del translation
    ###############################################################################
    # word2vec translation
    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector='word2vec')
    ranking = translation.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.word2vec.ranking')
    translation.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Word2Vec Translation')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del translation
    ###############################################################################
    # wordvec+elmo translation
    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector='word2vec+elmo')
    ranking = translation.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.word2vec_elmo.ranking')
    translation.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Word2Vec+ElMo Translation')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del translation
    ###############################################################################
    # cosine
    cosine = SemevalCosine()
    ranking = cosine.validate()

    path = os.path.join(EVALUATION_PATH, 'cosine.ranking')
    cosine.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Cosine')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del cosine
    ###############################################################################
    # align cosine
    aligncosine = SemevalSoftCosine(vector='alignments')
    ranking = aligncosine.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.align.ranking')
    aligncosine.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Align Cosine')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del aligncosine
    ###############################################################################
    # word2vec cosine
    softcosine = SemevalSoftCosine(vector='word2vec')
    ranking = softcosine.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.word2vec.ranking')
    softcosine.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Word2Vec SoftCosine')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del softcosine
    ###############################################################################
    # word2vec + elmo cosine
    softcosine = SemevalSoftCosine(vector='word2vec+elmo')
    ranking = softcosine.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.word2vec_elmo.ranking')
    softcosine.save(ranking, path)

    map_baseline, map_model = evaluate(ranking)
    print('Evaluation Word2Vec+ELMo SoftCosine')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    del softcosine