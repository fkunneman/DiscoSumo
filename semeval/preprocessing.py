__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': 'localhost', 'user': 'tcastrof'}
logger = logging.getLogger('tcpserver')

import json
import load
import os
import utils

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

DATA_PATH='data'
WRITE_TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
WRITE_DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

def run():
    props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

    logging.info('Load corpus', extra=d)
    trainset, devset = load.run()

    logging.info('Preparing development set...', extra=d)
    devset = utils.prepare_corpus(devset, corenlp=corenlp, props=props)
    json.dump(devset, open(WRITE_DEV_PATH, 'w'))

    logging.info('Preparing trainset...', extra=d)
    trainset = utils.prepare_corpus(trainset, corenlp=corenlp, props=props)
    json.dump(trainset, open(WRITE_TRAIN_PATH, 'w'))

    corenlp.close()

if __name__ == '__main__':
    run()