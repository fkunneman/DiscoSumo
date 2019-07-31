__author__='thiagocastroferreira'

import os

DATA_PATH='/roaming/tcastrof/quora/dataset'

def load(path):
    with open(path) as f:
        dataset = [w.split('\t') for w in f.read().split('\n')]

    head, dataset = dataset[0], dataset[1:-1]
    dataset = [dict(zip(head, q)) for q in dataset]
    return dataset

def run():
    path = os.path.join(DATA_PATH, 'train.tsv')
    trainset = load(path)

    path = os.path.join(DATA_PATH, 'dev.tsv')
    devset = load(path)

    path = os.path.join(DATA_PATH, 'test.tsv')
    testset = load(path)

    return trainset, devset, testset
