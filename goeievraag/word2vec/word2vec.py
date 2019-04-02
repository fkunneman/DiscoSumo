__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import json
import load
stopwords = load.load_stopwords()
import os
import string
punctuation = string.punctuation
import spacy

from gensim.models import Word2Vec


def run(question_path, answer_path, write_path, w_dim=300, window=10):
    nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
    documents = []

    questions = json.load(open(question_path))
    for question in questions:
        text = question['questiontext'] + ' '
        text += question['description']

        text = list(map(lambda token: str(token).lower(), nlp(text)))
        text = [w for w in text if w not in stopwords and w not in punctuation]
        documents.append(text)

    answers = json.load(open(answer_path))
    for answer in answers:
        text = answer['answertext']
        text = list(map(lambda token: str(token).lower(), nlp(text)))
        text = [w for w in text if w not in stopwords and w not in punctuation]
        documents.append(text)

    logging.info('Training...')
    fname = 'word2vec.' + str(w_dim) + '_' + str(window) + '.model'
    path = os.path.join(write_path, fname)
    model = Word2Vec(documents, size=w_dim, window=window, min_count=1, workers=10)
    model.save(path)

if __name__ == '__main__':
    QUESTIONS='../data/question_parsed.json'

    ANSWERS='../data/answer_parsed.json'

    logging.info('Loading corpus...')
    run(QUESTIONS, ANSWERS, '.')