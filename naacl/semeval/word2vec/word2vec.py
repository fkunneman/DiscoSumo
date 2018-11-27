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
QATAR_PATH='/roaming/tcastrof/semeval/dataset/unannotated/dump_QL_all_question_subject_body.txt.gz'
COMMENT_QATAR_PATH='/roaming/tcastrof/semeval/dataset/unannotated/dump_QL_all_comment_subject_body.txt.gz'

def load():
    with gzip.open(QATAR_PATH, 'rb') as f:
        questions = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    with gzip.open(COMMENT_QATAR_PATH, 'rb') as f:
        comments = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    return questions, comments

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
    path = '/roaming/tcastrof/semeval/word2vec'
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
            processes.append(pool.apply_async(parse, [i+1, chunk, 9010+i]))

        document = []
        for process in processes:
            doc = process.get()
            document.extend(doc)

        p.dump(document, open(fname, 'wb'))
        pool.close()
        pool.join()

        time.sleep(30)
    else:
        document = p.load(open(fname, 'rb'))

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