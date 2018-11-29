__author__='thiagocastroferreira'

import copy
import os

from quora_svm import QuoraSVM

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


def run_svm(model_, metric, stop, vector, evaluation_path, feature_path, alpha=0.7, sigma=0.3, gridsearch='brutal'):
    model = QuoraSVM(model=model_, features=metric, comment_features=metric, stop=stop, vector=vector, path=feature_path, sigma=sigma, alpha=alpha, gridsearch=gridsearch)
    result_dev = model.validate()

    y_real, y_pred, parameter_settings = result_dev
    f1score = f1_score(y_real, y_pred)
    accuracy = accuracy_score(y_real, y_pred)
    print('Evaluation: ', evaluation_path)
    print('Parameters:', parameter_settings)
    print('Accuracy: ', accuracy)
    print('F-Score: ', f1score)


if __name__ == '__main__':
    # SVM / REGRESSION
    # model_, metric, stop, vector, evaluation_path, feature_path, alpha=0.9, sigma=0.1
    # svm / softcosine / stop / word2vec+elmo
    path = 'svm.softcosine.stop.word2vec_elmo.ranking'
    feature_path = 'softcosine.stop.word2vec_elmo.features'
    run_svm(model_='svm', metric='softcosine,', stop=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path, sigma=0.3, alpha=0.7)
    ###############################################################################
    # regression / translation / stop / word2vec+elmo
    path = 'regression.softcosine.stop.word2vec_elmo.ranking'
    feature_path = 'softcosine.stop.word2vec_elmo.features'
    run_svm(model_='regression', metric='softcosine,', stop=True, vector='word2vec+elmo', evaluation_path=path, feature_path=feature_path, sigma=0.3, alpha=0.7)
    ################################################################################