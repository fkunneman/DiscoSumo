__author__='thiagocastroferreira'

import sys
sys.path.append('../')
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

import _pickle as p
import elmo.elmo as elmo
import json
import paths
import numpy as np
from nltk.corpus import stopwords
stop_ = set(stopwords.words('english'))
import os
import preprocessing
import re
import word2vec.word2vec as word2vec
import word2vec.fasttext as fasttext

from gensim import corpora

ALIGNMENTS_PATH=os.path.join(paths.ALIGNMENTS_PATH, 'align')
WORD2VEC_PATH=paths.WORD2VEC_PATH
FASTTEXT_PATH=paths.FASTTEXT_PATH
ELMO_PATH=paths.ELMO_PATH

ADDITIONAL_PATH=paths.ADDITIONAL_PATH
DATA_PATH=paths.DATA_PATH
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')
TEST2016_PATH=os.path.join(DATA_PATH, 'testset2016.data')
TEST2017_PATH=os.path.join(DATA_PATH, 'testset2017.data')

class Semeval():
    def __init__(self, stop=True, vector='', lowercase=True, punctuation=True, proctrain=True, elmo_layer='top', w2vdim=300):
        if not os.path.exists(DEV_PATH):
            preprocessing.run()

        self.lowercase = lowercase
        self.stop = stop
        self.punctuation = punctuation
        self.w2vdim = w2vdim
        self.proctrain = proctrain
        self.vector = vector
        self.elmo_layer = elmo_layer

        logging.info('Preparing test set 2016...')
        self.testset2016 = json.load(open(TEST2016_PATH))
        self.test2016data, _, _, _ = self.format_data(self.testset2016)

        logging.info('Preparing test set 2017...')
        self.testset2017 = json.load(open(TEST2017_PATH))
        self.test2017data, _, _, _ = self.format_data(self.testset2017)

        logging.info('Preparing development set...')
        self.devset = json.load(open(DEV_PATH))
        self.devdata, _, _, _ = self.format_data(self.devset)

        logging.info('Preparing trainset...')
        self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = self.format_data(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info)

        self.word2vec = None
        if 'word2vec' in self.vector:
            self.word2vec = word2vec.init_word2vec(lowercase=self.lowercase, punctuation=self.punctuation, stop=self.stop, dim=self.w2vdim)

        self.fasttext = None
        if 'fasttext' in self.vector:
            self.fasttext = fasttext.init_fasttext(lowercase=self.lowercase, punctuation=self.punctuation, stop=self.stop, dim=self.w2vdim)

        self.trainidx = self.trainelmo = self.devidx = self.develmo = self.test2016idx = self.test2016elmo = self.test2017idx = self.test2017elmo = None
        if 'elmo' in self.vector:
            self.trainidx, self.trainelmo, self.devidx, self.develmo, self.test2016idx, self.test2016elmo, self.test2017idx, self.test2017elmo = elmo.init_elmo(lowercase=self.lowercase, stop=self.stop, punctuation=self.punctuation, path=ELMO_PATH)

        self.alignments = self.init_alignments(ALIGNMENTS_PATH)

        # additional data
        self.init_additional()


    def init_alignments(self, path):
        if self.lowercase: path += '.lower'
        if self.stop: path += '.stop'
        if self.punctuation: path += '.punct'

        fname = os.path.join(path, 'model/lex.f2e')
        with open(fname) as f:
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


    def init_additional(self):
        path = ADDITIONAL_PATH + '.' + str(self.w2vdim)

        if self.proctrain:
            if self.lowercase: path += '.lower'
            if self.stop: path += '.stop'
            if self.punctuation: path += '.punct'
        fname = os.path.join(path, 'corpus.pickle')

        self.additional = p.load(open(fname, 'rb'))


    def encode(self, qid, q, elmoidx, elmovec):
        def w2v():
            emb = []
            for w in q:
                try:
                    emb.append(self.word2vec[w])
                except:
                    emb.append(self.w2vdim * [0])
            return emb

        def fasttext():
            emb = []
            for w in q:
                try:
                    emb.append(self.fasttext[w])
                except:
                    emb.append(self.w2vdim * [0])
            return emb

        def elmo():
            vec =  elmovec.get(str(elmoidx[qid]))
            if self.elmo_layer == 'average':
                vec = np.mean(vec, axis=0)
            elif self.elmo_layer == 'bottom':
                vec = vec[0]
            elif self.elmo_layer == 'middle':
                vec = vec[1]
            else:
                vec = vec[2]
            return vec

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
    def format_data(self, indexset):
        procset, vocabulary = {}, []

        vocquestions = []
        for i, qid in enumerate(indexset):
            procset[qid] = {}
            percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')

            question = indexset[qid]
            subj_q1_tree = question['subj_tree']
            q1_tree = question['tree']
            q1_pos = question['pos']
            q1_lemmas = question['lemmas_proc']
            q1_full = question['tokens']
            q1 = question['tokens']

            if self.lowercase:
                q1_lemmas = [w.lower() for w in q1_lemmas]
                q1_full = [w.lower() for w in q1_full]
                q1 = [w.lower() for w in q1]

            if self.punctuation:
                q1_lemmas = self.remove_punctuation(q1_lemmas)
                q1 = self.remove_punctuation(q1)

            if self.stop:
                q1_lemmas = self.remove_stopwords(q1_lemmas)
                q1 = self.remove_stopwords(q1)

            vocquestions.append(q1)
            vocabulary.extend(q1)

            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']
                subj_q2_tree = rel_question['subj_tree']
                q2_tree = rel_question['tree']
                q2_pos = rel_question['pos']
                q2_lemmas = rel_question['lemmas_proc']
                q2_full = rel_question['tokens']
                q2 = rel_question['tokens']

                if self.lowercase:
                    q2_lemmas = [w.lower() for w in q2_lemmas]
                    q2_full = [w.lower() for w in q2_full]
                    q2 = [w.lower() for w in q2]

                if self.punctuation:
                    q2_lemmas = self.remove_punctuation(q2_lemmas)
                    q2 = self.remove_punctuation(q2)

                if self.stop:
                    q2_lemmas = self.remove_stopwords(q2_lemmas)
                    q2 = self.remove_stopwords(q2)

                vocquestions.append(q2)
                vocabulary.extend(q2)

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
                    'ranking': int(rel_question['ranking']),
                    'comments': comments,
                    'label':label
                }

        vocabulary.append('UNK')
        vocabulary.append('eos')
        vocabulary = list(set(vocabulary))

        id2voc = {}
        for i, trigram in enumerate(vocabulary):
            id2voc[i] = trigram

        voc2id = dict(map(lambda x: (x[1], x[0]), id2voc.items()))

        vocabulary = corpora.Dictionary(vocquestions)
        return procset, voc2id, id2voc, vocabulary


    def format_pairwise_data(self, indexset):
        procset, vocabulary = {}, []

        vocquestions = []
        for i, qid in enumerate(indexset):
            procset[qid] = []
            percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
            print('Process: ', percentage, end='\r')

            question = indexset[qid]
            subj_q1_tree = question['subj_tree']
            q1_tree = question['tree']
            q1_pos = question['pos']
            q1_lemmas = question['lemmas_proc']
            q1_full = question['tokens']
            q1 = question['tokens']

            if self.lowercase:
                q1_lemmas = [w.lower() for w in q1_lemmas]
                q1_full = [w.lower() for w in q1_full]
                q1 = [w.lower() for w in q1]

            if self.punctuation:
                q1_lemmas = self.remove_punctuation(q1_lemmas)
                q1 = self.remove_punctuation(q1)

            if self.stop:
                q1_lemmas = self.remove_stopwords(q1_lemmas)
                q1 = self.remove_stopwords(q1)

            vocquestions.append(q1)
            vocabulary.extend(q1)

            duplicates = question['duplicates']
            for duplicate in duplicates:
                question2 = duplicate['rel_question']

                q2id = question2['id']
                subj_q2_tree = question2['subj_tree']
                q2_tree = question2['tree']
                q2_pos = question2['pos']
                q2_lemmas = question2['lemmas_proc']
                q2_full = question2['tokens']
                q2 = question2['tokens']
                q2ranking = int(question2['ranking'])
                q2relevance = question2['relevance']

                if self.lowercase:
                    q2_lemmas = [w.lower() for w in q2_lemmas]
                    q2_full = [w.lower() for w in q2_full]
                    q2 = [w.lower() for w in q2]

                if self.punctuation:
                    q2_lemmas = self.remove_punctuation(q2_lemmas)
                    q2 = self.remove_punctuation(q2)

                if self.stop:
                    q2_lemmas = self.remove_stopwords(q2_lemmas)
                    q2 = self.remove_stopwords(q2)

                vocquestions.append(q2)
                vocabulary.extend(q2)

                for duplicate2 in duplicates:
                    question3 = duplicate2['rel_question']
                    q3id = question3['id']
                    subj_q3_tree = question3['subj_tree']
                    q3_tree = question3['tree']
                    q3_pos = question3['pos']
                    q3_lemmas = question3['lemmas_proc']
                    q3_full = question3['tokens']
                    q3 = question3['tokens']
                    q3ranking = int(question3['ranking'])
                    q3relevance = question3['relevance']

                    if q2ranking < q3ranking and q2relevance == 'Irrelevant' and q3relevance != 'Irrelevant':
                        if self.lowercase:
                            q3_lemmas = [w.lower() for w in q3_lemmas]
                        q3_full = [w.lower() for w in q3_full]
                        q3 = [w.lower() for w in q3]

                        if self.punctuation:
                            q3_lemmas = self.remove_punctuation(q3_lemmas)
                            q3 = self.remove_punctuation(q3)

                        if self.stop:
                            q3_lemmas = self.remove_stopwords(q3_lemmas)
                            q3 = self.remove_stopwords(q3)

                        label = 1
                        procset[qid].append({
                            'q1_id': qid,
                            'q1': q1,
                            'q1_full': q1_full,
                            'q1_tree': q1_tree,
                            'subj_q1_tree': subj_q1_tree,
                            'q1_lemmas': q1_lemmas,
                            'q1_pos': q1_pos,
                            'q2_id': q2id,
                            'q2': q2,
                            'q2_full': q2_full,
                            'q2_tree': q2_tree,
                            'subj_q2_tree': subj_q2_tree,
                            'q2_lemmas': q2_lemmas,
                            'q2_pos': q2_pos,
                            'q3_id': q3id,
                            'q3': q3,
                            'q3_full': q3_full,
                            'q3_tree': q3_tree,
                            'subj_q3_tree': subj_q3_tree,
                            'q3_lemmas': q3_lemmas,
                            'q3_pos': q3_pos,
                            'label':label
                        })

                    elif q2ranking < q3ranking and q2relevance != 'Irrelevant' and q3relevance == 'Irrelevant':
                        if self.lowercase:
                            q3_lemmas = [w.lower() for w in q3_lemmas]
                        q3_full = [w.lower() for w in q3_full]
                        q3 = [w.lower() for w in q3]

                        if self.punctuation:
                            q3_lemmas = self.remove_punctuation(q3_lemmas)
                            q3 = self.remove_punctuation(q3)

                        if self.stop:
                            q3_lemmas = self.remove_stopwords(q3_lemmas)
                            q3 = self.remove_stopwords(q3)

                        label = 0
                        procset[qid].append({
                            'q1_id': qid,
                            'q1': q1,
                            'q1_full': q1_full,
                            'q1_tree': q1_tree,
                            'subj_q1_tree': subj_q1_tree,
                            'q1_lemmas': q1_lemmas,
                            'q1_pos': q1_pos,
                            'q2_id': q2id,
                            'q2': q2,
                            'q2_full': q2_full,
                            'q2_tree': q2_tree,
                            'subj_q2_tree': subj_q2_tree,
                            'q2_lemmas': q2_lemmas,
                            'q2_pos': q2_pos,
                            'q3_id': q3id,
                            'q3': q3,
                            'q3_full': q3_full,
                            'q3_tree': q3_tree,
                            'subj_q3_tree': subj_q3_tree,
                            'q3_lemmas': q3_lemmas,
                            'q3_pos': q3_pos,
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