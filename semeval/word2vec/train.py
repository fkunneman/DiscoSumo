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
QATAR_PATH='/home/tcastrof/Question/semeval/dataset/unannotated/dump_QL_all_question_subject_body.txt.gz'
COMMENT_QATAR_PATH='/home/tcastrof/Question/semeval/dataset/unannotated/dump_QL_all_comment_subject_body.txt.gz'

def load():
    with gzip.open(QATAR_PATH, 'rb') as f:
        questions = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    with gzip.open(COMMENT_QATAR_PATH, 'rb') as f:
        comments = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    return questions, comments

def parse_stop(document):
    props = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

    doc = []
    for i, sentence in enumerate(document):
        if i % 10000 == 0:
            percentage = str(round((float(i+1) / len(document)) * 100, 2)) + '%'
            print('Process: ', percentage)

        try:
            out = corenlp.annotate(re.sub(r'[^A-Za-z0-9]+', ' ' , sentence).strip(), properties=props)
            out = json.loads(out)

            tokens = []
            for snt in out['sentences']:
                words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens']) if w not in stop]
                tokens.extend(words)

            doc.append(tokens)
        except Exception as e:
            print('parsing error...')
            print(e)

    corenlp.close()
    return doc

def parse(thread_id, document, port, stop_=False, punct=False):
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
            # separate punctuation
            if punct:
                sentence = re.sub(r'([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])', ' ', sentence)
            else:
                sentence = re.sub(r'([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])', ' \1 ', sentence)
            out = corenlp.annotate(sentence.strip(), properties=props)
            out = json.loads(out)

            tokens = []
            for snt in out['sentences']:
                if stop_:
                    words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens']) if w not in stop]
                else:
                    words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens'])]
                tokens.extend(words)

            doc.append(tokens)
        except Exception as e:
            print('parsing error...')
            print(e)

    corenlp.close()
    time.sleep(5)
    return doc

def run(stop_=True, punct=True):
    fname = 'corpus_stop.pickle' if stop else 'corpus.pickle'
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
            processes.append(pool.apply_async(parse, [i+1, chunk, 9010+i, stop_, punct]))

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
    fname = 'word2vec_stop.model' if stop else 'word2vec.model'
    model = Word2Vec(document, size=300, window=10, min_count=10, workers=10)
    model.save(fname)

if __name__ == '__main__':
    logging.info('Loading corpus...')
    run(True, True)
    run(False, False)