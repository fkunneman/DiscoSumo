__author__='thiagocastroferreira'

import os
import semeval.load

TRAIN_PATH='train'
DEV_PATH='dev'

def run():
    def process(procset):
        indexes, sentences = [], []
        for i, qid in enumerate(procset):
            percentage = str(round((float(i+1) / len(procset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')
            question = trainset[qid]
            q1 = question['subject'] + ' ' + question['body']
            indexes.append(','.join([qid, '-', '-']))
            sentences.append(q1)

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2 = rel_question['subject']
                if rel_question['body']:
                    q2 += ' ' + rel_question['body']
                indexes.append(rel_question['id'])
                sentences.append(q2)
        return indexes, sentences

    trainset, devset = semeval.load.run()
    trainsnt, trainidx = process(trainset)

    if not os.path.exists(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)

    with open(os.path.join(TRAIN_PATH, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(trainsnt))

    with open(os.path.join(TRAIN_PATH, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in trainidx]))

    devsnt, devidx = process(trainset)

    if not os.path.exists(DEV_PATH):
        os.mkdir(DEV_PATH)

    with open(os.path.join(DEV_PATH, 'sentences.txt'), 'w') as f:
        f.write('\n'.join(devsnt))

    with open(os.path.join(DEV_PATH, 'index.txt'), 'w') as f:
        f.write('\n'.join([str(x) for x in devidx]))