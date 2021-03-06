__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/10/2018
Description:
    Script for extracting features for our ranking model
"""

import sys

sys.path.append('../')
import dynet as dy
import h5py
import nltk
import os
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from gensim.summarization import bm25
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
import lda
# from gensim.models import KeyedVectors
# from gensim.test.utils import datapath

from translation import *

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': 'localhost', 'user': 'tcastrof'}
logger = logging.getLogger('tcpserver')


WORD2VEC_PATH='/home/tcastrof/Question/DiscoSumo/semeval/word2vec/word2vec_stop.model'
GLOVE_PATH='/home/tcastrof/workspace/glove/glove.6B.300d.txt'
ELMO_PATH='elmo/'
TRANSLATION_PATH='translation/model/lex.f2e'

def lcsub(query, question):
    '''
    :param query:
    :param question:
    :return: longest common substring and size
    '''
    if len(query) == 0 or len(question) == 0:
        return 0, ''

    # initialize SequenceMatcher object with
    # input string
    seqMatch = SequenceMatcher(None,query,question)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(query), 0, len(question))

    # print longest substring
    if (match.size!=0):
        sub = query[match.a: match.a + match.size]
    else:
        sub = ''
    return (len(sub), sub)


def lcs(query, question):
    if len(query) == 0 or len(question) == 0:
        return 0, ''

    matrix = [["" for x in range(len(question))] for x in range(len(query))]
    for i in range(len(query)):
        for j in range(len(question)):
            if query[i] == question[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = query[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + query[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)

    cs = matrix[-1][-1]

    return len(cs), cs


def jaccard(query, question, tokenize=False):
    '''
    :param query:
    :param question:
    :param tokenize:
    :return: jaccard distance
    '''
    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
    query = set(query.split())
    question = set(question.split())

    if len(query) == 0 or len(question) == 0:
        return 0

    return float(len(query & question)) / len(query | question)


def containment_similarities(query, question, tokenize=False):
    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
    query = set(query.split())
    question = set(question.split())

    if len(query) == 0 or len(question) == 0:
        return 0

    return float(len(query & question)) / len(query)

####################################
### implementation from Aalto-LeTech
###
### before applying greedy string tiling:
### git clone --depth 1 https://github.com/Aalto-LeTech/greedy-string-tiling.git
### cd greedy-string-tiling
### pip install . hypothesis
###################################

def greedy_string_tiling(query, question, tokenize=False):

    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
        
    return sum([x[2] for x in match(query, '', question, '', 2)]) / len(query)


def dice(query, question, tokenize=False):
    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
    query = set(query.split())
    question = set(question.split())

    if len(query) == 0 or len(question) == 0:
        return 0

    return distance.dice(query, question)


def cosine(q1, q2, n=1):
    vectorizer = CountVectorizer(ngram_range=(n,n), stop_words='english')
    model = vectorizer.fit((q1,q2))
    vectors = vectorizer.transform([q1,q2])
    return cosine_similarity(vectors)[0,1]

def init_lda(traindata, n_topics=50):

    # set corpus
    logging.info('Setting lda')
    questions = []
    qs = [] 
    for row in traindata:
        qid, q = row['q1_id'], row['q1']
        if qid not in qs:
            questions.append(' '.join(q))
            qs.append(qid)
        qid, q = row['q2_id'], row['q2']
        if qid not in qs:
            questions.append(' '.join(q))
            qs.append(qid)

    # vectorize corpus
    vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
    vectors = vectorizer.fit_transform(questions)

    # train lda
    print('Training model')
    model = lda.LDA(n_topics=n_topics, random_state=0, n_iter=1500)
    model.fit(vectors)
    print('DONE.')

    return model, vectorizer

def init_translation(traindata, vocabulary, alpha, sigma):
    logging.info('Load background probabilities', extra=d)
    # TO DO: improve this
    questions = {}
    for trainrow in traindata:
        qid, q = trainrow['q1_id'], trainrow['q1']
        if qid not in questions:
            questions[qid] = q

        qid, q = trainrow['q2_id'], trainrow['q2']
        if qid not in questions:
            questions[qid] = q
    w_C = compute_w_C(questions, vocabulary)  # background lm
    logging.info('Load translation probabilities')
    t2w = translation_prob(TRANSLATION_PATH)  # translation probabilities
    translation = TRLM([], w_C, t2w, len(vocabulary), alpha=alpha, sigma=sigma)  # translation-based language model
    return translation


def init_bm25(traindata,devdata=False,testdata=False):

    def add_data(data, qs, index_dct, ind):
        for row in data:
            qid, q = row['q1_id'], row['q1']
            if qid not in qs:
                index_dct[qid] = ind
                ind += 1
                qs.append(q)
            qid, q = row['q2_id'], row['q2']
            if qid not in qs:
                index_dct[qid] = ind
                ind += 1
                qs.append(q)
        return qs, index_dct, ind

    # set corpus
    logging.info('Setting corpus')
    questions = []
    index_qid = {}
    index = 0

    questions, index_qid, index = add_data(traindata, questions, index_qid, index)
    if devdata:
        questions, index_qid, index = add_data(devdata, questions, index_qid, index)
    if testdata:
        questions, index_qid, index = add_data(testdata, questions, index_qid, index)
            
    # dct = Dictionary(questions)  # initialize a Dictionary
    # corpus = [dct.doc2bow(text) for text in questions]

    # set bm25 model
    logging.info('Initializing bm25 model', extra=d)
    model = bm25.BM25(questions)

    # get average idf
    logging.info('Calculating average idf')
    average_idf = sum(map(lambda k: float(model.idf[k]), model.idf.keys())) / len(model.idf.keys())

    # return model, average_idf, dct, index_qid
    return model, average_idf, index_qid


def init_elmo(stop=True):
    train_path = os.path.join(ELMO_PATH, 'train') if stop else os.path.join(ELMO_PATH, 'train_full')
    trainelmo = h5py.File(os.path.join(train_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(train_path, 'index.txt')) as f:
        trainidx = f.read().split('\n')
        trainidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(trainidx)])

    dev_path = os.path.join(ELMO_PATH, 'dev') if stop else os.path.join(ELMO_PATH, 'dev_full')
    develmo = h5py.File(os.path.join(dev_path, 'elmovectors.hdf5'), 'r')
    with open(os.path.join(dev_path, 'index.txt')) as f:
        devidx = f.read().split('\n')
        devidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(devidx)])
    return trainidx, trainelmo, devidx, develmo


def init_word2vec():
    # return KeyedVectors.load_word2vec_format(datapath(WORD2VEC_PATH), binary=True)
    return Word2Vec.load(WORD2VEC_PATH)


def encode(question, w2vec):
    emb = []
    for w in question:
        try:
            emb.append(w2vec[w.lower()])
        except:
            emb.append(w2vec['null'])
    return emb


def init_glove():
    tokens, embeddings = [], []
    with open(GLOVE_PATH) as f:
        for row in f.read().split('\n')[:-1]:
            _row = row.split()
            embeddings.append(np.array([float(x) for x in _row[1:]]))
            tokens.append(_row[0])

    # insert unk and eos token in the representations
    tokens.append('UNK')
    tokens.append('eos')
    id2voc = {}
    for i, token in enumerate(tokens):
        id2voc[i] = token

    voc2id = dict(map(lambda x: (x[1], x[0]), id2voc.items()))

    UNK = np.random.uniform(-0.1, 0.1, (300,))
    eos = np.random.uniform(-0.1, 0.1, (300,))
    embeddings.append(UNK)
    embeddings.append(eos)

    return np.array(embeddings), voc2id, id2voc


def glove_encode(question, glovevec, voc2id):
    emb = []
    for w in question:
        try:
            emb.append(glovevec[voc2id[w].lower()])
        except:
            emb.append(glovevec[voc2id['UNK']])
    return emb


def frobenius_norm(query_emb, question_emb):
    def dycosine(query_vec, question_vec):
        num = dy.transpose(query_vec) * question_vec
        dem1 = dy.sqrt(dy.transpose(query_vec) * query_vec)
        dem2 = dy.sqrt(dy.transpose(question_vec) * question_vec)
        dem = dem1 * dem2

        return dy.cdiv(num, dem)

    query_emb = list(map(lambda x: dy.inputTensor(x), list(query_emb)))
    question_emb = list(map(lambda x: dy.inputTensor(x), list(question_emb)))

    frobenius = 0.0
    for i in range(len(query_emb)):
        for j in range(len(question_emb)):
            cos = dy.rectify(dycosine(query_emb[i], question_emb[j])).value()
            frobenius += (cos**2)

    dy.renew_cg()
    return np.sqrt(frobenius)


class TreeKernel():
    def __init__(self, alpha=0, decay=1, ignore_leaves=True, smoothed=True):
        self.alpha = alpha
        self.decay = decay
        self.ignore_leaves = ignore_leaves
        self.smoothed = smoothed


    def __call__(self, q1_tree, q2_tree, q1_emb=[], q2_emb=[]):
        result = 0
        self.q1_emb = q1_emb
        self.q2_emb = q2_emb

        for node1 in q1_tree['nodes']:
            node1_type = q1_tree['nodes'][node1]['type']
            edgelen1 = len(q1_tree['edges'][node1])
            for node2 in q2_tree['nodes']:
                node2_type = q2_tree['nodes'][node2]['type']
                if 'terminal' not in [node1_type, node2_type]:
                    edgelen2 = len(q2_tree['edges'][node2])
                    delta = (self.decay**(edgelen1+edgelen2)) * self.__delta__(q1_tree, q2_tree, node1, node2)
                    result += delta

        return result


    def __delta__(self, tree1, tree2, root1, root2):
        if 'production' not in tree1['nodes'][root1]:
            tree1['nodes'][root1]['production'] = self.get_production(tree1, root1)
        if 'production' not in tree2['nodes'][root2]:
            tree2['nodes'][root2]['production'] = self.get_production(tree2, root2)

        production1 = tree1['nodes'][root1]['production']
        production2 = tree2['nodes'][root2]['production']
        result = 0
        if production1 == production2:
            node1_type = tree1['nodes'][root1]['type']
            node2_type = tree2['nodes'][root2]['type']
            if node1_type == 'preterminal' and node2_type == 'preterminal':
                if not self.smoothed:
                    result = 1
                else:
                    child1 = tree1['edges'][root1][0]
                    child2 = tree2['edges'][root2][0]

                    idx1 = tree1['nodes'][child1]['idx']
                    idx2 = tree2['nodes'][child2]['idx']
                    result = cosine_similarity([self.q1_emb[idx1]], [self.q2_emb[idx2]])[0][0]
            else:
                result = 1
                for i in range(len(tree1['edges'][root1])):
                    if result == 0:
                        break
                    child1 = tree1['edges'][root1][i]
                    child2 = tree2['edges'][root2][i]
                    result *= (self.alpha + self.__delta__(tree1, tree2, child1, child2))
        return result


    def similar_terminals(self, q1_tree, q2_tree):
        for node1 in q1_tree['nodes']:
            node1_type = q1_tree['nodes'][node1]['type']
            for node2 in q2_tree['nodes']:
                node2_type = q2_tree['nodes'][node2]['type']

                if node1_type == 'terminal' and node2_type == 'terminal':
                    w1 = q1_tree['nodes'][node1]['name'].replace('-rel', '').strip()
                    w2 = q2_tree['nodes'][node2]['name'].replace('-rel', '').strip()
                    lemma1 = q1_tree['nodes'][node1]['lemma']
                    lemma2 = q2_tree['nodes'][node2]['lemma']

                    if (w1 == w2) or (lemma1 == lemma2):
                        if '-rel' not in q1_tree['nodes'][node1]['name']:
                            q1_tree['nodes'][node1]['name'] += '-rel'
                        if '-rel' not in q2_tree['nodes'][node2]['name']:
                            q2_tree['nodes'][node2]['name'] += '-rel'

                        # fathers
                        prev_id1 = q1_tree['nodes'][node1]['parent']
                        if prev_id1 in q1_tree['nodes']:
                            if '-rel' not in q1_tree['nodes'][prev_id1]['name']:
                                q1_tree['nodes'][prev_id1]['name'] += '-rel'

                            prev_prev_id1 = q1_tree['nodes'][prev_id1]['parent']
                            if prev_prev_id1 in q1_tree['nodes']:
                                if '-rel' not in q1_tree['nodes'][prev_prev_id1]['name']:
                                    q1_tree['nodes'][prev_prev_id1]['name'] += '-rel'


                        prev_id2 = q2_tree['nodes'][node2]['parent']
                        if prev_id2 in q2_tree['nodes']:
                            if '-rel' not in q2_tree['nodes'][prev_id2]['name']:
                                q2_tree['nodes'][prev_id2]['name'] += '-rel'

                            prev_prev_id2 = q2_tree['nodes'][prev_id2]['parent']
                            if prev_prev_id2 in q2_tree['nodes']:
                                if '-rel' not in q2_tree['nodes'][prev_prev_id2]['name']:
                                    q2_tree['nodes'][prev_prev_id2]['name'] += '-rel'
        return q1_tree, q2_tree


    def get_production(self, tree, root):
        production = tree['nodes'][root]['name']
        node_type = tree['nodes'][root]['type']

        if node_type == 'nonterminal' or (node_type == 'preterminal' and not self.ignore_leaves):
            production += ' -> '
            for child in tree['edges'][root]:
                production += tree['nodes'][child]['name'] + ' '
        return production.strip()


    def __is_same_production__(self, tree1, tree2, root1, root2):
        production1 = tree1['nodes'][root1]['name'] + ' -> '
        for child in tree1['edges'][root1]:
            production1 += tree1['nodes'][child]['name'] + ' '

        production2 = tree2['nodes'][root2]['name'] + ' -> '
        for child in tree2['edges'][root2]:
            production2 += tree2['nodes'][child]['name'] + ' '

        if production1.strip() == production2.strip():
            return True
        else:
            return False


    def print_tree(self, root, tree, stree=''):
        if tree['nodes'][root]['type'] == 'terminal':
            stree += ' ' + tree['nodes'][root]['name']
        else:
            stree += '(' + tree['nodes'][root]['name'] + ' '

        for node in tree['edges'][root]:
            stree = self.print_tree(node, tree, stree) + ')'
        return stree


if __name__ == '__main__':
    question = 'Longest common subsequence long'
    query = 'Longest common substring long'

    start = time.time()
    print(lcs(re.split('(\W)', query), re.split('(\W)', question)))
    for i in range(2, 4):
        q1 = ''
        for gram in nltk.ngrams(query.split(), i):
            q1 += 'x'.join(gram) + ' '

        q2 = ''
        for gram in nltk.ngrams(question.split(), i):
            q2 += 'x'.join(gram) + ' '

        print(lcs(re.split('(\W)', q1), re.split('(\W)', q2)))
    end = time.time()
    print(end-start)
    start = time.time()
    print(lcsub(query, question))
    end = time.time()
    print(end-start)
    print(10 * '-')
    start = time.time()
    print(jaccard(query, question, True))
    end = time.time()
    print(end-start)
