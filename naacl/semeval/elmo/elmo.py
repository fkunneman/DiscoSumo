__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import h5py
import os
import json
import re

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
DATA_PATH='../data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')
TEST2016_PATH=os.path.join(DATA_PATH, 'testset2016.data')
TEST2017_PATH=os.path.join(DATA_PATH, 'testset2017.data')

def run(stop, write_train, write_dev, write_test2016, write_test2017):
    def parse(question):
        try:
            out = corenlp.annotate(question, properties=props)
            out = json.loads(out)

            tokens = []
            for snt in out['sentences']:
                tokens.extend(map(lambda x: x['originalText'], snt['tokens']))
        except:
            print('parsing error...')
            tokens = re.split(r'[\W]+', question)

        tokens = ' '.join(tokens)
        # treating empty documents to avoid error on allennlp
        if tokens.strip() == '':
            tokens = 'eos'
        return tokens

    def process(procset):
        indexes, sentences = [], []
        for i, qid in enumerate(procset):
            percentage = str(round((float(i+1) / len(procset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')
            question = procset[qid]
            q1 = ' '.join(question['tokens_proc']) if stop else ' '.join(question['tokens'])

            indexes.append(','.join([qid, '-', '-']))
            sentences.append(parse(q1))

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']

                q2 = ' '.join(rel_question['tokens_proc']) if stop else ' '.join(rel_question['tokens'])
                indexes.append(rel_question['id'])
                sentences.append(parse(q2))

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    q2 = ' '.join(rel_comment['tokens_proc']) if stop else ' '.join(rel_comment['tokens'])
                    indexes.append(rel_comment['id'])
                    sentences.append(parse(q2))
        return indexes, sentences

    props={'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

    # trainset, devset = load.run()

    # TRAINSET
    trainset = json.load(open(TRAIN_PATH))
    trainidx, trainsnt = process(trainset)

    if not os.path.exists(write_train):
        os.mkdir(write_train)

    with open(os.path.join(write_train, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(trainsnt))

    with open(os.path.join(write_train, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in trainidx]))

    # DEVSET
    devset = json.load(open(DEV_PATH))
    devidx, devsnt = process(devset)

    if not os.path.exists(write_dev):
        os.mkdir(write_dev)

    with open(os.path.join(write_dev, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(devsnt))

    with open(os.path.join(write_dev, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in devidx]))

    # TESTSET 2016
    testset2016 = json.load(open(TEST2016_PATH))
    test2016idx, test2016snt = process(testset2016)

    if not os.path.exists(write_test2016):
        os.mkdir(write_test2016)

    with open(os.path.join(write_test2016, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(test2016snt))

    with open(os.path.join(write_test2016, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in test2016idx]))

    # TESTSET 2017
    testset2017 = json.load(open(TEST2017_PATH))
    test2017idx, test2017snt = process(testset2017)

    if not os.path.exists(write_test2017):
        os.mkdir(write_test2017)

    with open(os.path.join(write_test2017, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(test2017snt))

    with open(os.path.join(write_test2017, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in test2017idx]))

    corenlp.close()

def init_elmo(stop, path):
    train_path = os.path.join(path, 'train') if stop else os.path.join(path, 'train_full')
    trainelmo = h5py.File(os.path.join(train_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(train_path, 'index.txt')) as f:
        trainidx = f.read().split('\n')
        trainidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(trainidx)])

    dev_path = os.path.join(path, 'dev') if stop else os.path.join(path, 'dev_full')
    develmo = h5py.File(os.path.join(dev_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(dev_path, 'index.txt')) as f:
        devidx = f.read().split('\n')
        devidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(devidx)])

    test2016_path = os.path.join(path, 'test2016') if stop else os.path.join(path, 'test2016_full')
    test2016elmo = h5py.File(os.path.join(test2016_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(test2016_path, 'index.txt')) as f:
        test2016idx = f.read().split('\n')
        test2016idx = dict([(qid.split(',')[0], i) for i, qid in enumerate(test2016idx)])

    test2017_path = os.path.join(path, 'test2017') if stop else os.path.join(path, 'test2017_full')
    test2017elmo = h5py.File(os.path.join(test2017_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(test2017_path, 'index.txt')) as f:
        test2017idx = f.read().split('\n')
        test2017idx = dict([(qid.split(',')[0], i) for i, qid in enumerate(test2017idx)])

    return trainidx, trainelmo, devidx, develmo, test2016idx, test2016elmo, test2017idx, test2017elmo

if __name__ == '__main__':
    WRITE_TRAIN_ELMO='/roaming/tcastrof/elmo/train'
    WRITE_DEV_ELMO='/roaming/tcastrof/elmo/dev'
    WRITE_TEST2016_ELMO='/roaming/tcastrof/elmo/test2016'
    WRITE_TEST2017_ELMO='/roaming/tcastrof/elmo/test2017'
    run(True, WRITE_TRAIN_ELMO, WRITE_DEV_ELMO, WRITE_TEST2016_ELMO, WRITE_TEST2017_ELMO)

    WRITE_TRAIN_ELMO='/roaming/tcastrof/elmo/train_full'
    WRITE_DEV_ELMO='/roaming/tcastrof/elmo/dev_full'
    WRITE_TEST2016_ELMO='/roaming/tcastrof/elmo/test2016_full'
    WRITE_TEST2017_ELMO='/roaming/tcastrof/elmo/test2017_full'
    run(False, WRITE_TRAIN_ELMO, WRITE_DEV_ELMO, WRITE_TEST2016_ELMO, WRITE_TEST2017_ELMO)