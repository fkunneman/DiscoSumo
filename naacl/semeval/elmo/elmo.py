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

def run(stop, write_train, write_dev):
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

    trainset = json.load(open(TRAIN_PATH))
    trainidx, trainsnt = process(trainset)

    if not os.path.exists(write_train):
        os.mkdir(write_train)

    with open(os.path.join(write_train, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(trainsnt))

    with open(os.path.join(write_train, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in trainidx]))

    devset = json.load(open(DEV_PATH))
    devidx, devsnt = process(devset)

    if not os.path.exists(write_dev):
        os.mkdir(write_dev)

    with open(os.path.join(write_dev, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(devsnt))

    with open(os.path.join(write_dev, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in devidx]))

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
    return trainidx, trainelmo, devidx, develmo

if __name__ == '__main__':
    WRITE_TRAIN_ELMO='train'
    WRITE_DEV_ELMO='dev'
    run(True, WRITE_TRAIN_ELMO, WRITE_DEV_ELMO)

    WRITE_TRAIN_ELMO='train_full'
    WRITE_DEV_ELMO='dev_full'
    run(False, WRITE_TRAIN_ELMO, WRITE_DEV_ELMO)