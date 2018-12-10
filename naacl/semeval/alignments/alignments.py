__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import json
import paths
import os
import re

from nltk.corpus import stopwords
stop_ = set(stopwords.words('english'))

DATA_PATH=paths.DATA_PATH
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

ALIGNMENTS_PATH=paths.ALIGNMENTS_PATH
if not os.path.exists(ALIGNMENTS_PATH):
    os.mkdir(ALIGNMENTS_PATH)

class SemevalAlignments():
    def __init__(self, lowercase=True, stop=True, punctuation=True):
        self.lowercase = lowercase
        self.stop = stop
        self.punctuation = punctuation

        self.devset = json.load(open(DEV_PATH))

        self.trainset = json.load(open(TRAIN_PATH))
        self.prepare()


    def remove_punctuation(self, tokens):
        return re.sub(r'[\W]+',' ', ' '.join(tokens)).strip().split()


    def remove_stopwords(self, tokens):
        return [w for w in tokens if w.lower() not in stop_]


    def prepare(self):
        corpus = []

        for i, q1id in enumerate(self.trainset):
            query = self.trainset[q1id]
            q1 = [w.lower() for w in query['tokens']] if self.lowercase else query['tokens']
            q1 = self.remove_punctuation(q1) if self.punctuation else q1
            q1 = self.remove_stopwords(q1) if self.stop else q1
            q1 = ' '.join(q1)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2 = [w.lower() for w in rel_question['tokens']] if self.lowercase else rel_question['tokens']
                q2 = self.remove_punctuation(q2) if self.punctuation else q2
                q2 = self.remove_stopwords(q2) if self.stop else q2
                q2 = ' '.join(q2)

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
                    q3 = [w.lower() for w in comment['tokens']] if self.lowercase else comment['tokens']
                    q3 = self.remove_punctuation(q3) if self.punctuation else q3
                    q3 = self.remove_stopwords(q3) if self.stop else q3
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

        path = os.path.join(ALIGNMENTS_PATH, 'align')
        if self.lowercase: path += '.lower'
        if self.stop: path += '.stop'
        if self.punctuation: path += '.punct'
        if not os.path.exists(path):
            os.mkdir(path)

        with open(os.path.join(path, 'semeval.de'), 'w') as f:
            f.write('\n'.join(map(lambda x: x['source'], corpus)))

        with open(os.path.join(path, 'semeval.en'), 'w') as f:
            f.write('\n'.join(map(lambda x: x['target'], corpus)))

if __name__ == '__main__':
    SemevalAlignments(lowercase=True, stop=True, punctuation=True)
    SemevalAlignments(lowercase=False, stop=True, punctuation=True)
    SemevalAlignments(lowercase=True, stop=False, punctuation=True)
    SemevalAlignments(lowercase=True, stop=True, punctuation=False)
    SemevalAlignments(lowercase=False, stop=False, punctuation=True)
    SemevalAlignments(lowercase=True, stop=False, punctuation=False)
    SemevalAlignments(lowercase=False, stop=True, punctuation=False)
    SemevalAlignments(lowercase=False, stop=False, punctuation=False)