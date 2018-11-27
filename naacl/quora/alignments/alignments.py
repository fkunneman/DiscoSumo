__author__='thiagocastroferreira'

import json
import os

DATA_PATH='/roaming/tcastrof/quora/dataset'
TRAIN_PATH=os.path.join(DATA_PATH, 'train.data')

ALIGNMENTS_PATH='/roaming/tcastrof/quora/alignments'
if not os.path.exists(ALIGNMENTS_PATH):
    os.mkdir(ALIGNMENTS_PATH)

class QuoraAlignments:
    def __init__(self):
        self.trainset = json.load(open(TRAIN_PATH))
        self.prepare()

    def prepare(self):
        corpus = []

        for pair in enumerate(self.trainset):
            q1 = ' '.join(pair['tokens1'])
            q2 = ' '.join(pair['tokens2'])

            corpus.append({
                'source': q1,
                'target': q2
            })
            corpus.append({
                'source': q2,
                'target': q1
            })

        with open('quora.de', 'w') as f:
            f.write('\n'.join(map(lambda x: x['source'], corpus)))

        with open('quora.en', 'w') as f:
            f.write('\n'.join(map(lambda x: x['target'], corpus)))

if __name__ == '__main__':
    QuoraAlignments()