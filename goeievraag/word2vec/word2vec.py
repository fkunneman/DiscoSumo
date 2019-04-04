__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import os
import string
punctuation = string.punctuation

from gensim.models import Word2Vec

def run(documents, write_path, w_dim=300, window=10):
    logging.info('Training...')
    fname = 'word2vec.' + str(w_dim) + '_' + str(window) + '.model'
    path = os.path.join(write_path, fname)
    model = Word2Vec(documents, size=w_dim, window=window, min_count=3, workers=10)
    model.save(path)