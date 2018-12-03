__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import h5py
import os
import json
import re
from nltk.corpus import stopwords
stop_ = set(stopwords.words('english'))

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
DATA_PATH='../data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')
TEST2016_PATH=os.path.join(DATA_PATH, 'testset2016.data')
TEST2017_PATH=os.path.join(DATA_PATH, 'testset2017.data')

def remove_punctuation(tokens):
    return re.sub(r'[\W]+',' ', ' '.join(tokens)).strip().split()

def remove_stopwords(tokens):
    return [w for w in tokens if w.lower() not in stop_]

def run(stop, lowercase, punctuation, write_train, write_dev, write_test2016, write_test2017):
    def process(procset):
        indexes, sentences = [], []
        for i, qid in enumerate(procset):
            percentage = str(round((float(i+1) / len(procset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')
            question = procset[qid]
            q1 = [w.lower() for w in question['tokens']] if lowercase else question['tokens']
            q1 = remove_punctuation(q1) if punctuation else q1
            q1 = remove_stopwords(q1) if stop else q1

            indexes.append(','.join([qid, '-', '-']))
            sentences.append(q1)

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2 = [w.lower() for w in rel_question['tokens']] if lowercase else rel_question['tokens']
                q2 = remove_punctuation(q2) if punctuation else q2
                q2 = remove_stopwords(q2) if stop else q2
                indexes.append(rel_question['id'])
                sentences.append(q2)

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    q3 = [w.lower() for w in rel_comment['tokens']] if lowercase else rel_comment['tokens']
                    q3 = remove_punctuation(q3) if punctuation else q3
                    q3 = remove_stopwords(q3) if stop else q3
                    if len(q3) == 0:
                        q3 = ['eos']
                    indexes.append(rel_comment['id'])
                    sentences.append(q3)
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

def init_elmo(stop, lowercase, punctuation, path):
    train_path = os.path.join(path, 'train')
    dev_path = os.path.join(path, 'dev')
    test2016_path = os.path.join(path, 'test2016')
    test2017_path = os.path.join(path, 'test2017')

    if lowercase:
        train_path += '.lower'
        dev_path += '.lower'
        test2016_path += '.lower'
        test2017_path += '.lower'
    if stop:
        train_path += '.stop'
        dev_path += '.stop'
        test2016_path += '.stop'
        test2017_path += '.stop'
    if punctuation:
        train_path += '.punct'
        dev_path += '.punct'
        test2016_path += '.punct'
        test2017_path += '.punct'

    trainelmo = h5py.File(os.path.join(train_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(train_path, 'index.txt')) as f:
        trainidx = f.read().split('\n')
        trainidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(trainidx)])

    develmo = h5py.File(os.path.join(dev_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(dev_path, 'index.txt')) as f:
        devidx = f.read().split('\n')
        devidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(devidx)])

    test2016elmo = h5py.File(os.path.join(test2016_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(test2016_path, 'index.txt')) as f:
        test2016idx = f.read().split('\n')
        test2016idx = dict([(qid.split(',')[0], i) for i, qid in enumerate(test2016idx)])

    test2017elmo = h5py.File(os.path.join(test2017_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(test2017_path, 'index.txt')) as f:
        test2017idx = f.read().split('\n')
        test2017idx = dict([(qid.split(',')[0], i) for i, qid in enumerate(test2017idx)])

    return trainidx, trainelmo, devidx, develmo, test2016idx, test2016elmo, test2017idx, test2017elmo

if __name__ == '__main__':
    path = '/roaming/tcastrof/semeval/elmo/'
    train_path = os.path.join(path, 'train.lower.stop.punct')
    dev_path = os.path.join(path, 'dev.lower.stop.punct')
    test2016_path = os.path.join(path, 'test2016.lower.stop.punct')
    test2017_path = os.path.join(path, 'test2017.lower.stop.punct')
    run(lowercase=True, stop=True, punctuation=True,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train.lower.stop')
    dev_path = os.path.join(path, 'dev.lower.stop')
    test2016_path = os.path.join(path, 'test2016.lower.stop')
    test2017_path = os.path.join(path, 'test2017.lower.stop')
    run(lowercase=True, stop=True, punctuation=False,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train.lower.punct')
    dev_path = os.path.join(path, 'dev.lower.punct')
    test2016_path = os.path.join(path, 'test2016.lower.punct')
    test2017_path = os.path.join(path, 'test2017.lower.punct')
    run(lowercase=True, stop=False, punctuation=True,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train.stop.punct')
    dev_path = os.path.join(path, 'dev.stop.punct')
    test2016_path = os.path.join(path, 'test2016.stop.punct')
    test2017_path = os.path.join(path, 'test2017.stop.punct')
    run(lowercase=False, stop=True, punctuation=True,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train.lower')
    dev_path = os.path.join(path, 'dev.lower')
    test2016_path = os.path.join(path, 'test2016.lower')
    test2017_path = os.path.join(path, 'test2017.lower')
    run(lowercase=True, stop=False, punctuation=False,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train.stop')
    dev_path = os.path.join(path, 'dev.stop')
    test2016_path = os.path.join(path, 'test2016.stop')
    test2017_path = os.path.join(path, 'test2017.stop')
    run(lowercase=False, stop=True, punctuation=False,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train.punct')
    dev_path = os.path.join(path, 'dev.punct')
    test2016_path = os.path.join(path, 'test2016.punct')
    test2017_path = os.path.join(path, 'test2017.punct')
    run(lowercase=False, stop=False, punctuation=True,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)

    train_path = os.path.join(path, 'train')
    dev_path = os.path.join(path, 'dev')
    test2016_path = os.path.join(path, 'test2016')
    test2017_path = os.path.join(path, 'test2017')
    run(lowercase=False, stop=False, punctuation=False,
        write_train=train_path, write_dev=dev_path, write_test2016=test2016_path, write_test2017=test2017_path)