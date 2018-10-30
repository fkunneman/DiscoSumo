__author__='thiagocastroferreira'

import json
import gzip
import re
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import time
from multiprocessing import Pool

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
QATAR_PATH='/home/tcastrof/Question/semeval/dataset/unannotated/dump_QL_all_question_subject_body.txt.gz'
COMMENT_QATAR_PATH='/home/tcastrof/Question/semeval/dataset/unannotated/dump_QL_all_comment_subject_body.txt.gz'

def load():
    with gzip.open(QATAR_PATH, 'rb') as f:
        questions = [s.replace('\r\\', ' ').replace('\n', ' ').strip() for s in f.read().decode('utf-8').split('<COMMENTS_SEPARATOR>')]

    return questions

def parse(thread_id, document, port):
    props = {'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', port=port)

    time.sleep(5)

    doc, stopdoc, treedoc = [], [], []
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
            sentence = re.sub(r'([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])', ' \1 ', sentence)
            out = corenlp.annotate(sentence.strip(), properties=props)
            out = json.loads(out)

            trees = '(SENTENCES '
            tokens, tokens_stop = [], []
            for snt in out['sentences']:
                words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens']) if w not in stop]
                tokens_stop.extend(words)
                words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens'])]
                tokens.extend(words)
                trees += snt['parse'].replace('\n', '') + ' '
            trees = trees.strip()
            trees += ')'

            doc.append(' '.join(tokens))

            tokens_stop = re.sub(r'[^A-Za-z0-9]+', ' ', ' '.join(tokens_stop)).strip()
            stopdoc.append(tokens_stop)

            treedoc.append(trees)
        except Exception as e:
            print('parsing error...')
            print(e)

    corenlp.close()
    time.sleep(5)
    return doc, stopdoc, treedoc

def run():
    questions = load()
    documents = questions

    THREADS = 20
    n = int(len(documents) / THREADS)
    chunks = [documents[i:i+n] for i in range(0, len(documents), n)]

    pool = Pool(processes=len(chunks))

    logging.info('Parsing corpus...')
    processes = []
    for i, chunk in enumerate(chunks):
        print('Process id: ', i+1, 'Doc length:', len(chunk))
        processes.append(pool.apply_async(parse, [i+1, chunk, 9010+i]))

    document, stopdocument, treedocument = [], [], []
    for process in processes:
        doc, stopdoc, treedoc = process.get()
        document.extend(doc)
        stopdocument.extend(stopdoc)
        treedocument.extend(treedoc)

    pool.close()
    pool.join()

    with open('question.txt', 'w') as f:
        f.write('\n'.join(document))

    with open('question.stop.txt', 'w') as f:
        f.write('\n'.join(stopdocument))

    with open('question.tree', 'w') as f:
        f.write('\n'.join(treedocument))

if __name__ == '__main__':
    logging.info('Loading corpus...')
    run()