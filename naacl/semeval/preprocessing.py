__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import copy
import json
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import load
import os
import re

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

DATA_PATH='data'
WRITE_TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
WRITE_DEV_PATH=os.path.join(DATA_PATH, 'devset.data')
WRITE_TEST2016_PATH=os.path.join(DATA_PATH, 'testset2016.data')
WRITE_TEST2017_PATH=os.path.join(DATA_PATH, 'testset2017.data')

def parse(question, corenlp, props):
    tokens, lemmas, pos = [], [], []
    out = None
    try:
        out = corenlp.annotate(question, properties=props)
        parsed = json.loads(out)

        trees = '(SENTENCES '
        for snt in parsed['sentences']:
            tokens.extend(map(lambda x: x['originalText'], snt['tokens']))
            lemmas.extend(map(lambda x: x['lemma'], snt['tokens']))
            pos.extend(map(lambda x: x['pos'], snt['tokens']))
            trees += snt['parse'].replace('\n', '') + ' '
        trees = trees.strip()
        trees += ')'
    except:
        print('parsing error...')
        print(out)
        props_ = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
        out = corenlp.annotate(question, properties=props_)
        parsed = json.loads(out)

        tokens = []
        for snt in parsed['sentences']:
            words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens'])]
            tokens.extend(words)
        trees = '()'
    return ' '.join(tokens), trees, ' '.join(lemmas), pos

def preprocess(indexset, corenlp, props):
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')
        question = indexset[qid]
        q1 = copy.copy(question['subject'])
        tokens, question['subj_tree'], lemmas, pos = parse(q1, corenlp, props)

        q1 = question['subject'] + ' ' + question['body']
        tokens, question['tree'], lemmas, pos = parse(q1, corenlp, props)
        question['tokens'] = [w for w in tokens.lower().split()]
        q1 = re.sub(r'[\W]+',' ', tokens).strip()
        q1 = [w for w in q1.lower().split() if w not in stop]
        question['tokens_proc'] = q1

        question['lemmas'] = [w for w in lemmas.lower().split()]
        q1_lemmas = re.sub(r'[\W]+',' ', lemmas.lower()).strip()
        q1_lemmas = [w for w in q1_lemmas.split() if w not in stop]
        question['lemmas_proc'] = q1_lemmas

        question['pos'] = pos

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2 = copy.copy(rel_question['subject'])
            tokens, rel_question['subj_tree'], lemmas, pos = parse(q2, corenlp, props)

            q2 = copy.copy(rel_question['subject'])
            if rel_question['body']:
                q2 += ' ' + rel_question['body']
            tokens, rel_question['tree'], lemmas, pos = parse(q2, corenlp, props)
            rel_question['tokens'] = [w for w in tokens.lower().split()]
            q2 = re.sub(r'[\W]+',' ', tokens).strip()
            q2 = [w for w in q2.lower().split() if w not in stop]
            rel_question['tokens_proc'] = q2

            rel_question['lemmas'] = [w for w in lemmas.lower().split()]
            q2_lemmas = re.sub(r'[\W]+',' ', lemmas.lower()).strip()
            q2_lemmas = [w for w in q2_lemmas.split() if w not in stop]
            rel_question['lemmas_proc'] = q2_lemmas

            rel_question['pos'] = pos

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                q3 = rel_comment['text']
                tokens, rel_comment['tree'], lemmas, pos = parse(q3, corenlp, props)
                rel_comment['tokens'] = [w for w in tokens.lower().split()]
                q3 = re.sub(r'[\W]+',' ', tokens).strip()
                q3 = [w for w in q3.lower().split() if w not in stop]
                rel_comment['tokens_proc'] = q3

                rel_comment['lemmas'] = [w for w in lemmas.lower().split()]
                q3_lemmas = re.sub(r'[\W]+',' ', lemmas.lower()).strip()
                q3_lemmas = [w for w in q3_lemmas.split() if w not in stop]
                rel_comment['lemmas_proc'] = q3_lemmas

                rel_comment['pos'] = pos

    return indexset

def run():
    props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', memory='8g')

    logging.info('Load corpus')
    trainset, devset, testset2016, testset2017 = load.run()

    logging.info('Preparing test set 2016...')
    testset2016 = preprocess(testset2016, corenlp=corenlp, props=props)
    json.dump(testset2016, open(WRITE_TEST2016_PATH, 'w'))

    logging.info('Preparing test set 2017...')
    testset2017 = preprocess(testset2017, corenlp=corenlp, props=props)
    json.dump(testset2017, open(WRITE_TEST2017_PATH, 'w'))

    logging.info('Preparing development set...')
    devset = preprocess(devset, corenlp=corenlp, props=props)
    json.dump(devset, open(WRITE_DEV_PATH, 'w'))

    logging.info('Preparing trainset...')
    trainset = preprocess(trainset, corenlp=corenlp, props=props)
    json.dump(trainset, open(WRITE_TRAIN_PATH, 'w'))

    corenlp.close()

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    if not os.path.exists(WRITE_DEV_PATH):
        run()