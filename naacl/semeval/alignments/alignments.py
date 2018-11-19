__author__='thiagocastroferreira'

import sys
sys.path.append('../')

from semeval import Semeval

def init_alignments(path):
    with open(path) as f:
        doc = list(map(lambda x: x.split(), f.read().split('\n')))

    alignments = {}
    for row in doc[:-1]:
        t = row[0]
        if t[0] not in alignments:
            alignments[t[0]] = {}
        if t not in alignments[t[0]]:
            alignments[t[0]][t] = {}

        w = row[1]
        if w[0] not in alignments[t[0]][t]:
            alignments[t[0]][t][w[0]] = {}

        prob = float(row[2])
        alignments[t[0]][t][w[0]][w] = prob
    return alignments

class SemevalAlignments(Semeval):
    def __init__(self):
        Semeval.__init__(self)
        self.prepare()

    def prepare(self):
        corpus = []

        for i, q1id in enumerate(self.trainset):
            query = self.trainset[q1id]
            q1 = query['tokens']

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = rel_question['tokens']

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