__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import h5py
import os
import json

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
DATA_PATH='/roaming/tcastrof/quora/dataset'
TRAIN_PATH=os.path.join(DATA_PATH, 'train.data')
DEV_PATH=os.path.join(DATA_PATH, 'dev.data')
TEST_PATH=os.path.join(DATA_PATH, 'test.data')

ELMO_PATH='/roaming/tcastrof/quora/elmo'
if not os.path.exists(ELMO_PATH):
    os.mkdir(ELMO_PATH)

def run(stop, write_train, write_dev, write_test):
    def process(procset):
        indexes, sentences = [], []
        for i, pair in enumerate(procset):
            percentage = str(round((float(i+1) / len(procset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')
            qid = pair['qid1']
            q1 = pair['tokens_proc1'] if stop else  pair['tokens1']

            indexes.append(','.join([qid, '-', '-']))
            sentences.append(q1)

            qid = pair['qid2']
            q2 = pair['tokens_proc2'] if stop else  pair['tokens2']

            indexes.append(','.join([qid, '-', '-']))
            sentences.append(q2)
        return indexes, sentences

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
    testset = json.load(open(TEST_PATH))
    test2016idx, test2016snt = process(testset)

    if not os.path.exists(write_test):
        os.mkdir(write_test)

    with open(os.path.join(write_test, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(test2016snt))

    with open(os.path.join(write_test, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in test2016idx]))

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

    test_path = os.path.join(path, 'test') if stop else os.path.join(path, 'test_full')
    testelmo = h5py.File(os.path.join(test_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(test_path, 'index.txt')) as f:
        testidx = f.read().split('\n')
        testidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(testidx)])

    return trainidx, trainelmo, devidx, develmo, testidx, testelmo

if __name__ == '__main__':
    WRITE_TRAIN_ELMO=os.path.join(ELMO_PATH, 'train')
    WRITE_DEV_ELMO=os.path.join(ELMO_PATH, 'dev')
    WRITE_TEST_ELMO=os.path.join(ELMO_PATH, 'test')
    run(True, WRITE_TRAIN_ELMO, WRITE_DEV_ELMO, WRITE_TEST_ELMO)

    WRITE_TRAIN_ELMO=os.path.join(ELMO_PATH, 'train_full')
    WRITE_DEV_ELMO=os.path.join(ELMO_PATH, 'dev_full')
    WRITE_TEST_ELMO=os.path.join(ELMO_PATH, 'test_full')
    run(False, WRITE_TRAIN_ELMO, WRITE_DEV_ELMO, WRITE_TEST_ELMO)