__author__='thiagocastroferreira'

import json
import os

DATA_PATH='../data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class SemevalAlignments():
    def __init__(self):
        self.devset = json.load(open(DEV_PATH))

        self.trainset = json.load(open(TRAIN_PATH))
        self.prepare()

    def prepare(self):
        corpus = []

        for i, q1id in enumerate(self.trainset):
            query = self.trainset[q1id]
            q1 = ' '.join(query['tokens'])

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = ' '.join(rel_question['tokens'])

                corpus.append({
                    'source': q1,
                    'target': q2
                })
                corpus.append({
                    'source': q2,
                    'target': q1
                })

                for comment in duplicate['rel_comments']:
                    q3id = comment['id']
                    q3 = comment['tokens']
                    if len(q3) == 0:
                        q3 = ['eos']

                    q3 = ' '.join(q3)
                    corpus.append({
                        'source': q1,
                        'target': q3
                    })
                    corpus.append({
                        'source': q3,
                        'target': q1
                    })

        with open('semeval.de', 'w') as f:
            f.write('\n'.join(map(lambda x: x['source'], corpus)))

        with open('semeval.en', 'w') as f:
            f.write('\n'.join(map(lambda x: x['target'], corpus)))

if __name__ == '__main__':
    SemevalAlignments()