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
from gensim.models import FastText
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
    path = '/roaming/tcastrof/semeval/word2vec'
    if not os.path.exists(path):
        os.mkdir(path)

    if lowercase and punctuation and stop:
        fname = os.path.join(path, 'corpus.lower.stop.punct.pickle')
    elif lowercase and punctuation and not stop:
        fname = os.path.join(path, 'corpus.lower.punct.pickle')
    elif lowercase and not punctuation and stop:
        fname = os.path.join(path, 'corpus.lower.stop.pickle')
    elif not lowercase and punctuation and stop:
        fname = os.path.join(path, 'corpus.stop.punct.pickle')
    elif lowercase and not punctuation and not stop:
        fname = os.path.join(path, 'corpus.lower.pickle')
    elif not lowercase and not punctuation and stop:
        fname = os.path.join(path, 'corpus.stop.pickle')
    elif not lowercase and punctuation and not stop:
        fname = os.path.join(path, 'corpus.punct.pickle')
    else:
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
    if lowercase and punctuation and stop:
        fname = os.path.join(path, 'fasttext.lower.stop.punct.pickle')
    elif lowercase and punctuation and not stop:
        fname = os.path.join(path, 'fasttext.lower.punct.pickle')
    elif lowercase and not punctuation and stop:
        fname = os.path.join(path, 'fasttext.lower.stop.pickle')
    elif not lowercase and punctuation and stop:
        fname = os.path.join(path, 'fasttext.stop.punct.pickle')
    elif lowercase and not punctuation and not stop:
        fname = os.path.join(path, 'fasttext.lower.pickle')
    elif not lowercase and not punctuation and stop:
        fname = os.path.join(path, 'fasttext.stop.pickle')
    elif not lowercase and punctuation and not stop:
        fname = os.path.join(path, 'fasttext.punct.pickle')
    else:
        fname = os.path.join(path, 'fasttext.pickle')
    model = FastText(document, size=300, window=10, min_count=1, workers=10, iter=5)
    model.save(fname)

def init_fasttext(lowercase=True, punctuation=True, stop=True):
    path = '/roaming/tcastrof/semeval/word2vec'
    if lowercase and punctuation and stop:
        fname = os.path.join(path, 'fasttext.lower.stop.punct.pickle')
    elif lowercase and punctuation and not stop:
        fname = os.path.join(path, 'fasttext.lower.punct.pickle')
    elif lowercase and not punctuation and stop:
        fname = os.path.join(path, 'fasttext.lower.stop.pickle')
    elif not lowercase and punctuation and stop:
        fname = os.path.join(path, 'fasttext.stop.punct.pickle')
    elif lowercase and not punctuation and not stop:
        fname = os.path.join(path, 'fasttext.lower.pickle')
    elif not lowercase and not punctuation and stop:
        fname = os.path.join(path, 'fasttext.stop.pickle')
    elif not lowercase and punctuation and not stop:
        fname = os.path.join(path, 'fasttext.punct.pickle')
    else:
        fname = os.path.join(path, 'fasttext.pickle')
    return FastText.load(fname)


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