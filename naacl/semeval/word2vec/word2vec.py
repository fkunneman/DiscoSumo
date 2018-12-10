__author__='thiagocastroferreira'

import sys
sys.path.append('../')
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
import paths
from gensim.models import Word2Vec
from multiprocessing import Pool

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=paths.STANFORD_PATH
QATAR_PATH=paths.QATAR_PATH
COMMENT_QATAR_PATH=paths.COMMENT_QATAR_PATH

def load():
    with gzip.open(QATAR_PATH, 'rb') as f:
        questions = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    with gzip.open(COMMENT_QATAR_PATH, 'rb') as f:
        comments = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    return questions, comments

def remove_punctuation(tokens):
    return re.sub(r'[\W]+',' ', ' '.join(tokens)).strip().split()

def remove_stopwords(tokens):
    return [w for w in tokens if w.lower() not in stop]

def parse(thread_id, document, port, lowercase=True, punctuation=True, stop=True):
    props = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', port=port)

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
                if lowercase:
                    words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens'])]
                else:
                    words = [w for w in map(lambda x: x['originalText'], snt['tokens'])]
                words = remove_punctuation(words) if punctuation else words
                words = remove_stopwords(words) if stop else words

                tokens.extend(words)

            doc.append(tokens)
        except Exception as e:
            print('parsing error...')
            print(e)

    corenlp.close()
    return doc

def train(lowercase=True, punctuation=True, stop=True):
    path = paths.WORD2VEC_DIR
    path = os.path.join(path, 'word2vec')
    if lowercase: path += '.lower'
    if stop: path += '.stop'
    if punctuation: path += '.punct'
    if not os.path.exists(path):
        os.mkdir(path)

    fname = os.path.join(path, 'corpus.pickle')
    if not os.path.exists(fname):
        questions, comments = load()
        documents = questions + comments

        THREADS = 25
        n = int(len(documents) / THREADS)
        chunks = [documents[i:i+n] for i in range(0, len(documents), n)]

        pool = Pool(processes=len(chunks))

        logging.info('Parsing corpus...')
        processes = []
        for i, chunk in enumerate(chunks):
            print('Process id: ', i+1, 'Doc length:', len(chunk))
            processes.append(pool.apply_async(parse, [i+1, chunk, 9010+i, lowercase, punctuation, stop]))

        document = []
        for process in processes:
            doc = process.get()
            document.extend(doc)

        p.dump(document, open(fname, 'wb'))
        pool.close()
        pool.join()
    else:
        document = p.load(open(fname, 'rb'))

    logging.info('Training...')
    fname = os.path.join(path, 'word2vec.model')
    model = Word2Vec(document, size=300, window=10, min_count=1, workers=10, iter=5)
    model.save(fname)

def init_word2vec(lowercase=True, punctuation=True, stop=True):
    path = paths.WORD2VEC_DIR
    path = os.path.join(path, 'word2vec')
    if lowercase: path += '.lower'
    if stop: path += '.stop'
    if punctuation: path += '.punct'

    fname = os.path.join(path, 'word2vec.model')
    return Word2Vec.load(fname)


def encode(question, w2vec):
    emb = []
    for w in question:
        try:
            emb.append(w2vec[w])
        except:
            emb.append(300 * [0])
    return emb

if __name__ == '__main__':
    logging.info('Loading corpus...')
    train(lowercase=True, stop=True, punctuation=True)
    train(lowercase=False, stop=True, punctuation=True)
    train(lowercase=True, stop=False, punctuation=True)
    train(lowercase=True, stop=True, punctuation=False)
    train(lowercase=False, stop=False, punctuation=True)
    train(lowercase=True, stop=False, punctuation=False)
    train(lowercase=False, stop=True, punctuation=False)
    train(lowercase=False, stop=False, punctuation=False)