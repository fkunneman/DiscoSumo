__author__='thiagocastroferreira'

import json
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import load
import os
import re

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

DATA_PATH='/roaming/tcastrof/quora/dataset'
WRITE_TRAIN_PATH=os.path.join(DATA_PATH, 'train.data')
WRITE_DEV_PATH=os.path.join(DATA_PATH, 'dev.data')
WRITE_TEST_PATH=os.path.join(DATA_PATH, 'test.data')

def parse(question, corenlp):
    props_ = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    out = corenlp.annotate(question, properties=props_)
    parsed = json.loads(out)

    tokens = []
    for snt in parsed['sentences']:
        words = [w for w in map(lambda x: x['originalText'].lower(), snt['tokens'])]
        tokens.extend(words)

    tokens = [w.lower() for w in tokens]
    tokens_proc = re.sub(r'[\W]+', ' ', ' '.join(tokens)).strip()
    tokens_proc = [w for w in tokens_proc.lower().split() if w not in stop]
    return tokens, tokens_proc

def preprocess(indexset, corenlp, set_='train'):
    pairs = []
    questions = {}
    errors = 0
    for pair in indexset:
        try:
            if set_ == 'test':
                question = pair['question1']
                tokens, tokens_proc = parse(question, corenlp)
                pair['tokens1'] = tokens
                pair['tokens_proc1'] = tokens_proc

                question = pair['question2']
                tokens, tokens_proc = parse(question, corenlp)
                pair['tokens2'] = tokens
                pair['tokens_proc2'] = tokens_proc
            else:
                qid1 = pair['qid1']
                if qid1 not in questions:
                    question = pair['question1']

                    tokens, tokens_proc = parse(question, corenlp)
                    pair['tokens1'] = tokens
                    pair['tokens_proc1'] = tokens_proc

                    questions[qid1] = {'question': question, 'tokens': tokens, 'tokens_proc': tokens_proc}
                else:
                    pair['tokens1'] = questions[qid1]['tokens']
                    pair['tokens_proc1'] = questions[qid1]['tokens_proc']

                qid2 = pair['qid2']
                if qid2 not in questions:
                    question = pair['question2']

                    tokens, tokens_proc = parse(question, corenlp)
                    pair['tokens2'] = tokens
                    pair['tokens_proc2'] = tokens_proc

                    questions[qid2] = {'question': question, 'tokens': tokens, 'tokens_proc': tokens_proc}
                else:
                    pair['tokens2'] = questions[qid2]['tokens']
                    pair['tokens_proc2'] = questions[qid2]['tokens_proc']

            pairs.append(pair)
        except:
            errors += 1
            print('Number of errors: ', errors)
    return pairs

def run():
    props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27', memory='8g')

    logging.info('Load corpus')
    trainset, devset, testset = load.run()

    logging.info('Preparing test set...')
    testset2017 = preprocess(testset, corenlp=corenlp, set_='test')
    json.dump(testset2017, open(WRITE_TEST_PATH, 'w'))

    logging.info('Preparing development set...')
    devset = preprocess(devset, corenlp=corenlp, set_='dev')
    json.dump(devset, open(WRITE_DEV_PATH, 'w'))

    logging.info('Preparing trainset...')
    trainset = preprocess(trainset, corenlp=corenlp, set_='train')
    json.dump(trainset, open(WRITE_TRAIN_PATH, 'w'))

    corenlp.close()

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    if not os.path.exists(WRITE_DEV_PATH):
        run()