__author__='thiagocastroferreira'

import json
import gzip

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

from gensim.models import Word2Vec

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
QATAR_PATH='/home/tcastrof/Question/semeval/dataset/unannotated/dump_QL_all_question_subject_body.txt.gz'
COMMENT_QATAR_PATH='/home/tcastrof/Question/semeval/dataset/unannotated/dump_QL_all_comment_subject_body.txt.gz'

def load():
    with gzip.open(QATAR_PATH, 'rb') as f:
        questions = [s.replace('\r\\', '').strip() for s in f.read().decode('utf-8').split('\n')]

    with gzip.open(COMMENT_QATAR_PATH, 'rb') as f:
        comments = [s.replace('\r\\', '').strip() for s in f.read().decode('utf-8').split('\n')]

    return questions, comments

def parse(document):
    props = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

    doc = []
    for i, snt in enumerate(document):
        percentage = str(round((float(i+1) / len(document)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')
        if snt.strip() not in ['', '<SUBJECT_BODY_SEPARATOR>', '<COMMENTS_SEPARATOR>']:
            out = corenlp.annotate(snt, properties=props)
            out = json.loads(out)

            tokens = []
            for snt in out['sentences']:
                tokens.extend(map(lambda x: x['originalText'], snt['tokens']))

            doc.append(tokens)

    corenlp.close()
    return doc

if __name__ == '__main__':
    logging.info('Loading corpus...')
    questions, comments = load()
    logging.info('Parsing corpus...')
    document = parse(questions + comments)
    logging.info('Training...')
    model = Word2Vec(document, size=300, window=5, min_count=50, workers=10)
    model.save('word2vec.model')