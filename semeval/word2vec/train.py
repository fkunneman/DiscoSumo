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

def parse(document):
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

if __name__ == '__main__':
    logging.info('Loading corpus...')
    questions, comments = load()
    documents = questions + comments

    THREADS = 10
    pool = Pool(processes=THREADS)
    n = int(len(documents) / THREADS)
    chunks = [documents[i:i+n] for i in range(0, len(documents), n)]

    logging.info('Parsing corpus...')
    processes = []
    for i, chunk in enumerate(chunks):
        processes.append(pool.apply_async(parse, [chunk]))

    document = []
    for process in processes:
        doc = process.get()
        document.extend(doc)

    p.dump(document, open('corpus.pickle', 'wb'))

    logging.info('Training...')
    model = Word2Vec(document, size=300, window=5, min_count=50, workers=10)
    model.save('word2vec.model')