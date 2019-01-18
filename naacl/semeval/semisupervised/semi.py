__author__='thiagocastroferreira'

import sys
sys.path.append('../')
sys.path.append('../../')
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import _pickle as p
import json
import paths
from nltk.corpus import stopwords
stop_ = set(stopwords.words('english'))
import os
import preprocessing
import re
import word2vec.word2vec as word2vec

ALIGNMENTS_PATH=os.path.join(paths.ALIGNMENTS_PATH, 'align')
WORD2VEC_PATH=paths.WORD2VEC_PATH
FASTTEXT_PATH=paths.FASTTEXT_PATH

ADDITIONAL_PATH=paths.ADDITIONAL_PATH
DATA_PATH=paths.DATA_PATH
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')
TEST2016_PATH=os.path.join(DATA_PATH, 'testset2016.data')
TEST2017_PATH=os.path.join(DATA_PATH, 'testset2017.data')

class Semi:
    def __init__(self, stop=True, lowercase=True, punctuation=True, w2v_dim=300):
        if not os.path.exists(DEV_PATH):
            preprocessing.run()

        self.w2v_dim = w2v_dim
        self.lowercase = lowercase
        self.stop = stop
        self.punctuation = punctuation

        logging.info('Preparing test set 2016...')
        self.testset2016 = json.load(open(TEST2016_PATH))
        self.test2016data = self.format_data(self.testset2016)

        logging.info('Preparing test set 2017...')
        self.testset2017 = json.load(open(TEST2017_PATH))
        self.test2017data = self.format_data(self.testset2017)

        logging.info('Preparing development set...')
        self.devset = json.load(open(DEV_PATH))
        self.devdata = self.format_data(self.devset)

        logging.info('Preparing trainset...')
        self.trainset = json.load(open(TRAIN_PATH))
        self.traindata = self.format_data(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info)

        self.word2vec = word2vec.init_word2vec(lowercase=self.lowercase, punctuation=self.punctuation, stop=self.stop, w2v_dim=self.w2v_dim)

        # additional data
        self.init_additional()


    def init_additional(self):
        path = ADDITIONAL_PATH

        path += '.' + str(self.w2v_dim)
        if self.lowercase: path += '.lower'
        if self.stop: path += '.stop'
        if self.punctuation: path += '.punct'
        fname = os.path.join(path, 'corpus.pickle')

        self.additional = p.load(open(fname, 'rb'))


    def encode(self, q):
        emb = []
        for w in q:
            try:
                emb.append(self.word2vec[w])
            except:
                emb.append(self.w2v_dim * [0])
        return emb


    # utilities
    def format_data(self, indexset):
        procset, vocabulary = {}, []

        vocquestions = []
        for i, qid in enumerate(indexset):
            procset[qid] = {}
            percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')

            question = indexset[qid]
            q1_full = question['tokens']
            q1 = question['tokens']

            if self.lowercase:
                q1_full = [w.lower() for w in q1_full]
                q1 = [w.lower() for w in q1]

            if self.punctuation:
                q1 = self.remove_punctuation(q1)

            if self.stop:
                q1 = self.remove_stopwords(q1)

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                q2_full = rel_question['tokens']
                q2 = rel_question['tokens']

                if self.lowercase:
                    q2_full = [w.lower() for w in q2_full]
                    q2 = [w.lower() for w in q2]

                if self.punctuation:
                    q2 = self.remove_punctuation(q2)

                if self.stop:
                    q2 = self.remove_stopwords(q2)

                # Related questions to augment the corpus
                comments = []

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    q3 = rel_comment['tokens']
                    if self.lowercase:
                        q3 = [w.lower() for w in q3]

                    if self.punctuation:
                        q3 = self.remove_punctuation(q3)

                    if self.stop:
                        q3 = self.remove_stopwords(q3)
                    vocquestions.append(q3)
                    vocabulary.extend(q3)

                    comments.append({
                        'id': rel_comment['id'],
                        'tokens': q3,
                    })

                label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    label = 1
                procset[qid][q2id] = {
                    'q1_id': qid,
                    'q1': q1,
                    'q1_full': q1_full,
                    'q2_id': rel_question['id'],
                    'q2': q2,
                    'q2_full': q2_full,
                    'comments': comments,
                    'label':label
                }

        return procset


    def save(self, ranking, path, parameter_settings):
        with open(path, 'w') as f:
            f.write(parameter_settings)
            f.write('\n')
            for q1id in ranking:
                for row in ranking[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))


    def remove_punctuation(self, tokens):
        return re.sub(r'[\W]+',' ', ' '.join(tokens)).strip().split()


    def remove_stopwords(self, tokens):
        return [w for w in tokens if w.lower() not in stop_]