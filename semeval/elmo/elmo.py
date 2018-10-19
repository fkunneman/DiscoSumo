__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import os
import load
import json
import re

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
DATA_PATH='../data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

WRITE_TRAIN_ELMO='train'
WRITE_DEV_ELMO='dev'

def run():
    def parse(question):
        try:
            out = corenlp.annotate(question, properties=props)
            out = json.loads(out)

            tokens = []
            for snt in out['sentences']:
                tokens.extend(map(lambda x: x['originalText'], snt['tokens']))
        except:
            print('parsing error...')
            tokens = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)

        tokens = ' '.join(tokens)
        # treating empty documents to avoid error on allennlp
        if tokens.strip() == '':
            tokens = 'eos'
        return tokens

    def process(procset):
        indexes, sentences = [], []
        for i, qid in enumerate(procset):
            percentage = str(round((float(i+1) / len(procset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')
            question = procset[qid]
            q1 = ' '.join(question['tokens_proc'])
            indexes.append(','.join([qid, '-', '-']))
            sentences.append(parse(q1))

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']

                q2 = ' '.join(rel_question['tokens_proc'])
                indexes.append(rel_question['id'])
                sentences.append(parse(q2))

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    q2 = ' '.join(rel_comment['tokens_proc'])
                    indexes.append(rel_comment['id'])
                    sentences.append(parse(q2))
        return indexes, sentences

    props={'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

    # trainset, devset = load.run()

    trainset = json.load(open(TRAIN_PATH))
    trainidx, trainsnt = process(trainset)

    if not os.path.exists(WRITE_TRAIN_ELMO):
        os.mkdir(WRITE_TRAIN_ELMO)

    with open(os.path.join(WRITE_TRAIN_ELMO, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(trainsnt))

    with open(os.path.join(WRITE_TRAIN_ELMO, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in trainidx]))

    devset = json.load(open(DEV_PATH))
    devidx, devsnt = process(devset)

    if not os.path.exists(WRITE_DEV_ELMO):
        os.mkdir(WRITE_DEV_ELMO)

    with open(os.path.join(WRITE_DEV_ELMO, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(devsnt))

    with open(os.path.join(WRITE_DEV_ELMO, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in devidx]))

    corenlp.close()

if __name__ == '__main__':
    run()