__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import _pickle as p
import elmo.elmo as elmo
import json
import numpy as np
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import preprocessing
import word2vec.word2vec as word2vec
import word2vec.fasttext as fasttext

from gensim import corpora

ALIGNMENTS_PATH='alignments/model/lex.f2e'
WORD2VEC_PATH='word2vec/word2vec.model'
FASTTEXT_PATH='word2vec/fasttext.model'
ELMO_PATH='elmo/'

ADDITIONAL_PATH= 'word2vec/corpus.pickle'
DATA_PATH='/home/tcastrof/Question/DiscoSumo/naacl/semeval/data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class Semeval():
    def __init__(self, vector=''):
        if not os.path.exists(DEV_PATH):
            preprocessing.run()

        self.vector = vector
        logging.info('Preparing development set...')
        self.devset = json.load(open(DEV_PATH))
        self.devdata, self.voc2id, self.id2voc, self.vocabulary = self.format_data(self.devset, parts=('dev'))

        logging.info('Preparing trainset...')
        self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = self.format_data(self.trainset, parts=('train1', 'train2'))
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info)

        self.word2vec = None
        if 'word2vec' in self.vector:
            self.word2vec = word2vec.init_word2vec(WORD2VEC_PATH)

        self.fasttext = None
        if 'fasttext' in self.vector:
            self.fasttext = fasttext.init_fasttext(FASTTEXT_PATH)

        self.trainidx = self.trainelmo = self.devidx = self.develmo = None
        self.fulltrainidx = self.fulltrainelmo = self.fulldevidx = self.fulldevelmo = None
        if 'elmo' in self.vector:
            self.trainidx, self.trainelmo, self.devidx, self.develmo = elmo.init_elmo(True, ELMO_PATH)
            self.fulltrainidx, self.fulltrainelmo, self.fulldevidx, self.fulldevelmo = elmo.init_elmo(False, ELMO_PATH)

        self.alignments = self.init_alignments(ALIGNMENTS_PATH)

        # additional data
        self.additional = p.load(open(ADDITIONAL_PATH, 'rb'))


    def init_alignments(self, path):
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

    def encode(self, qid, q, elmoidx, elmovec, encoding):
        def w2v():
            emb = []
            for w in q:
                try:
                    emb.append(self.word2vec[w.lower()])
                except:
                    emb.append(300 * [0])
            return emb

        def fasttext():
            emb = []
            for w in q:
                try:
                    emb.append(self.fasttext[w.lower()])
                except:
                    emb.append(300 * [0])
            return emb

        def elmo():
            return elmovec.get(str(elmoidx[qid]))

        if self.vector == 'word2vec':
            return w2v()
        elif self.vector == 'fasttext':
            return fasttext()
        elif self.vector == 'elmo':
            return elmo()
        elif self.vector == 'fasttext+elmo':
            w2vemb = fasttext()
            elmoemb = elmo()
            return [np.concatenate([w2vemb[i], elmoemb[i]]) for i in range(len(w2vemb))]
        elif self.vector == 'word2vec+elmo':
            w2vemb = w2v()
            elmoemb = elmo()
            return [np.concatenate([w2vemb[i], elmoemb[i]]) for i in range(len(w2vemb))]
        return 0


    # utilities
    def format_data(self, indexset, parts):
        procset, vocabulary = [], []

        vocquestions = []
        for i, qid in enumerate(indexset):
            if indexset[qid]['set'] in parts:
                percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
                print('Process: ', percentage, end='\r')

                question = indexset[qid]
                subj_q1_tree = question['subj_tree']
                q1_tree = question['tree']
                q1_pos = question['pos']
                q1_lemmas = question['lemmas_proc']
                q1_full = question['tokens']
                q1 = question['tokens_proc']

                vocquestions.append(q1)
                vocabulary.extend(q1)

                duplicates = question['duplicates']
                for duplicate in duplicates:
                    rel_question = duplicate['rel_question']
                    subj_q2_tree = rel_question['subj_tree']
                    q2_tree = rel_question['tree']
                    q2_pos = rel_question['pos']
                    q2_lemmas = rel_question['lemmas_proc']
                    q2_full = rel_question['tokens']
                    q2 = rel_question['tokens_proc']
                    vocquestions.append(q2)
                    vocabulary.extend(q2)

                    # Related questions to augment the corpus
                    comments = []

                    rel_comments = duplicate['rel_comments']
                    for rel_comment in rel_comments:
                        q3 = rel_comment['tokens_proc']
                        vocquestions.append(q3)
                        vocabulary.extend(q3)

                        comments.append({
                            'id': rel_comment['id'],
                            'tokens': q3,
                        })

                    label = 0
                    if rel_question['relevance'] != 'Irrelevant':
                        label = 1
                    procset.append({
                        'q1_id': qid,
                        'q1': q1,
                        'q1_full': q1_full,
                        'q1_tree': q1_tree,
                        'subj_q1_tree': subj_q1_tree,
                        'q1_lemmas': q1_lemmas,
                        'q1_pos': q1_pos,
                        'q2_id': rel_question['id'],
                        'q2': q2,
                        'q2_full': q2_full,
                        'q2_tree': q2_tree,
                        'subj_q2_tree': subj_q2_tree,
                        'q2_lemmas': q2_lemmas,
                        'q2_pos': q2_pos,
                        'comments': comments,
                        'label':label
                    })

        vocabulary.append('UNK')
        vocabulary.append('eos')
        vocabulary = list(set(vocabulary))

        id2voc = {}
        for i, trigram in enumerate(vocabulary):
            id2voc[i] = trigram

        voc2id = dict(map(lambda x: (x[1], x[0]), id2voc.items()))

        vocabulary = corpora.Dictionary(vocquestions)
        return procset, voc2id, id2voc, vocabulary


    def save(self, ranking, path):
        with open(path, 'w') as f:
            for q1id in ranking:
                for row in ranking[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))

