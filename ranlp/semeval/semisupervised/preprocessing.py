__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import paths
import os
import json
import gzip
import re
from nltk.corpus import stopwords
stop_ = set(stopwords.words('english'))
import string
punctuation = string.punctuation
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import time
from multiprocessing import Pool

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=paths.STANFORD_PATH
QATAR_PATH=paths.QATAR_PATH
COMMENT_QATAR_PATH=paths.COMMENT_QATAR_PATH
SEMI_PATH=paths.SEMI_PATH

if not os.path.exists(SEMI_PATH):
    os.mkdir(SEMI_PATH)

def load():
    with gzip.open(QATAR_PATH, 'rb') as f:
        questions = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    return questions

def parse(thread_id, document, port, lower=True, stop=False, punct=True):
    props = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', port=port, memory='8g', timeout=5500)

    doc, procdoc = [], []
    print('Thread id: ', thread_id, 'Doc length:', len(document))
    for i, q in enumerate(document):
        idx, question = q
        if i % 1000 == 0:
            percentage = str(round((float(i+1) / len(document)) * 100, 2)) + '%'
            print('Thread id: ', thread_id, 'Port:', port, 'Process: ', percentage)

        try:
            # remove urls
            question = re.sub(r'^https?:\/\/.*[\r\n]*', 'url', question)
            # remove html
            question = re.sub(r'<.*?>', ' . ', question)
            # separate punctuation
            question = ' '.join(re.split(r'([!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~])', question))
            out_ = corenlp.annotate(question.strip(), properties=props)
            out = json.loads(out_)

            tokens, tokens_proc = [], []
            for snt in out['sentences']:
                # proc tokens
                words = list(map(lambda x: x['originalText'].lower(), snt['tokens']))
                words = [w.lower() for w in words] if lower else words
                words = [w for w in words if w not in stop_] if stop else words
                words = [w for w in words if w not in punctuation] if punct else words
                tokens_proc.extend(words)

                # original tokens
                words = [w for w in map(lambda x: x['originalText'], snt['tokens'])]
                tokens.extend(words)
                tokens.append('<SENTENCE>')

            doc.append((idx, ' '.join(tokens)))

            tokens_proc = ' '.join(tokens_proc).strip()
            procdoc.append((idx, tokens_proc))
        except Exception as e:
            print('parsing error...')
            print(e)
            print(out_)
            print(10 * '-')

            doc.append((idx, ''))
            procdoc.append((idx, ''))

    corenlp.close()
    return doc, procdoc

def parse_tree(thread_id, document, port):
    props = {'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', port=port, memory='8g', timeout=5500)

    error = 0
    doc, stopdoc, treedoc = [], [], []
    print('Thread id: ', thread_id, 'Doc length:', len(document))
    for i, q in enumerate(document):
        idx, question = q
        if i % 100 == 0:
            percentage = str(round((float(i+1) / len(document)) * 100, 2)) + '%'
            perror = str(round((float(error) / len(document)) * 100, 2)) + '%'
            print('Thread id: ', thread_id, 'Port:', port, 'Process: ', percentage, 'Error: ', perror)

        try:
            trees = '(SENTENCES '
            for sentence in question.split('<SENTENCE>'):
                out_ = corenlp.annotate(sentence.strip(), properties=props)
                out = json.loads(out_)

                for snt in out['sentences']:
                    trees += snt['parse'].replace('\n', '') + ' '
                trees = trees.strip()
            trees += ')'
            treedoc.append(trees)
        except Exception as e:
            error += 1
            treedoc.append(' ')

    corenlp.close()
    return treedoc

def run():
    questions = load()
    # indexing
    questions = [(i, q) for i, q in enumerate(questions)]

    THREADS = 20
    n = int(len(questions) / THREADS)
    chunks = [questions[i:i+n] for i in range(0, len(questions), n)]

    pool = Pool(processes=len(chunks))

    logging.info('Parsing corpus...')
    processes = []
    for i, chunk in enumerate(chunks):
        print('Process id: ', i+1, 'Doc length:', len(chunk))
        processes.append(pool.apply_async(parse, [i + 1, chunk, 9010 + i]))

    document, procdocument = [], []
    for process in processes:
        doc, procdoc = process.get()
        document.extend(doc)
        procdocument.extend(procdoc)

    pool.close()
    pool.join()

    with open(os.path.join(SEMI_PATH, 'index.txt'), 'w') as f:
        idx = [str(q[0]) for q in document]
        f.write('\n'.join(idx))

    with open(os.path.join(SEMI_PATH, 'question.txt'), 'w') as f:
        questions = [q[1] for q in document]
        f.write('\n'.join(questions))

    with open(os.path.join(SEMI_PATH, 'question.proc.txt'), 'w') as f:
        questions = [q[1] for q in procdocument]
        f.write('\n'.join(questions))

    time.sleep(60)

    # Parse the trees
    n = int(len(document) / THREADS)
    chunks = [document[i:i+n] for i in range(0, len(document), n)]
    pool = Pool(processes=len(chunks))
    logging.info('Parsing corpus for tree...')
    processes = []
    for i, chunk in enumerate(chunks):
        print('Process id: ', i+1, 'Doc length:', len(chunk))
        processes.append(pool.apply_async(parse_tree, [i + 1, chunk, 9100 + i]))

    treedocument = []
    for process in processes:
        treedoc = process.get()
        treedocument.extend(treedoc)

    pool.close()
    pool.join()

    with open(os.path.join(SEMI_PATH, 'question.tree'), 'w') as f:
        f.write('\n'.join(treedocument))

if __name__ == '__main__':
    logging.info('Loading corpus...')
    run()