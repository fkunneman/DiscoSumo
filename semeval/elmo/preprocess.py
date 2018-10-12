__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import os
import load
import json
import re

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'
TRAIN_PATH='train'
DEV_PATH='dev'

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
        return ' '.join(tokens)

    def process(procset):
        indexes, sentences = [], []
        for i, qid in enumerate(procset):
            percentage = str(round((float(i+1) / len(procset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')
            question = procset[qid]
            q1 = question['subject'] + ' ' + question['body']
            indexes.append(','.join([qid, '-', '-']))
            sentences.append(parse(q1))

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2 = rel_question['subject']
                if rel_question['body']:
                    q2 += ' ' + rel_question['body']
                indexes.append(rel_question['id'])
                sentences.append(parse(q2))
        return indexes, sentences

    props={'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

    trainset, devset = load.run()
    trainidx, trainsnt = process(trainset)

    if not os.path.exists(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)

    with open(os.path.join(TRAIN_PATH, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(trainsnt))

    with open(os.path.join(TRAIN_PATH, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in trainidx]))

    devidx, devsnt = process(devset)

    if not os.path.exists(DEV_PATH):
        os.mkdir(DEV_PATH)

    with open(os.path.join(DEV_PATH, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(devsnt))

    with open(os.path.join(DEV_PATH, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in devidx]))

    corenlp.close()

if __name__ == '__main__':
    run()