__author__='thiagocastroferreira'

import _pickle as p
import json
import gzip
import re
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import os
import time
from gensim.models import Word2Vec
from multiprocessing import Pool

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

DATA_PATH='/roaming/tcastrof/quora/dataset'
TRAIN_PATH=os.path.join(DATA_PATH, 'train.data')

def load():
    trainset = json.load(open(TRAIN_PATH))
    questions = []
    for pair in enumerate(trainset):
        questions.append(pair['tokens1'])
        questions.append(pair['tokens2'])
    return questions

def parse(thread_id, document, port):
    props = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', port=port)

    time.sleep(5)

    doc = []
    print('Thread id: ', thread_id, 'Doc length:', len(document))
    for i, sentence in enumerate(document):
        if i % 10000 == 0:
            percentage = str(round((float(i+1) / len(document)) * 100, 2)) + '%'
            print('Thread id: ', thread_id, 'Port:', port, 'Process: ', percentage)

        try:
            # remove urls
            sentence = re.sub(r'^https?:\/\/.*[\r\n]*', 'url', sentence)
            # remove html
            sentence = re.sub(r'<.*?>', 'html', sentence)
            out = corenlp.annotate(sentence.strip(), properties=props)
            out = json.loads(out)

            tokens = []
            for snt in out['sentences']:
                words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens'])]
                tokens.extend(words)

            doc.append(tokens)
        except Exception as e:
            print('parsing error...')
            print(e)

    corenlp.close()
    time.sleep(5)
    return doc

def train():
    path = '/roaming/tcastrof/quora/word2vec'
    if not os.path.exists(path):
        os.mkdir(path)

    document = load()
    logging.info('Training...')
    fname = os.path.join(path, 'word2vec.model')
    model = Word2Vec(document, size=300, window=10, min_count=1, workers=10)
    model.save(fname)

def init_word2vec(path):
    return Word2Vec.load(path)


def encode(question, w2vec):
    emb = []
    for w in question:
        try:
            emb.append(w2vec[w.lower()])
        except:
            emb.append(300 * [0])
    return emb

if __name__ == '__main__':
    logging.info('Loading corpus...')
    train()