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
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import bm25
from gensim.corpora import Dictionary

from translation import *

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

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

def greedy_string_tiling(query, question, tokenize=False):
    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
    query = query.split()
    question = question.split()

    if len(query) == 0 or len(question) == 0:
        return 0

    # if py>3.0, nonlocal is better
    class markit:
        a=[0]
        minlen=1
    markit.a=[0]*len(query)
    markit.minlen=1

    #output char index
    out=[]

    # To find the max length substr (index)
    # apos is the position of a[0] in origin string
    def maxsub(a,b,apos=0,lennow=0):
        if (len(a) == 0 or len(b) == 0):
            return []
        if (a[0]==b[0] and markit.a[apos]!=1 ):
            return [apos]+maxsub(a[1:],b[1:],apos+1,lennow=lennow+1)
        elif (a[0]!=b[0] and lennow>0):
            return []
        return max(maxsub(a, b[1:],apos), maxsub(a[1:], b,apos+1), key=len)

    while True:
        findmax=maxsub(query,question,0,0)
        if (len(findmax)<markit.minlen):
            break
        else:
            for i in findmax:
                markit.a[i]=1
            out+=findmax
    gst = [ query[i] for i in out ]
    return len(gst) / len(query)

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
    vectorizer = CountVectorizer(ngram_range=(n,n))
    vectors = vectorizer.fit_transform([q1,q2])
    return cosine_similarity(vectors)[0,1]

def init_translation(traindata, vocabulary, alpha, sigma):
    logging.info('Load background probabilities')
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
    print('Setting corpus')
    questions = []
    index_qid = {}
    index = 0

    questions, index_qid, index = add_data(traindata, questions, index_qid, index)
    if devdata:
        questions, index_qid, index = add_data(devdata, questions, index_qid, index)
    if testdata:
        questions, index_qid, index = add_data(testdata, questions, index_qid, index)
            
    dct = Dictionary(questions)  # initialize a Dictionary
    corpus = [dct.doc2bow(text) for text in questions]

    # set bm25 model
    logging.info('Initializing bm25 model')
    model = bm25.BM25(corpus)

    # get average idf
    print('Calculating average idf')
    average_idf = sum(map(lambda k: float(model.idf[k]), model.idf.keys())) / len(model.idf.keys())

    return model, average_idf, dct, index_qid

def init_elmo():
    trainelmo = h5py.File(os.path.join(ELMO_PATH, 'train', 'elmovectors.hdf5'), 'r')
    with open(os.path.join(ELMO_PATH, 'train', 'index.txt')) as f:
        trainidx = f.read().split('\n')
        trainidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(trainidx)])

    develmo = h5py.File(os.path.join(ELMO_PATH, 'dev', 'elmovectors.hdf5'), 'r')
    with open(os.path.join(ELMO_PATH, 'dev', 'index.txt')) as f:
        devidx = f.read().split('\n')
        devidx = dict([(qid.split(',')[0], i) for i, qid in enumerate(devidx)])
    return trainidx, trainelmo, devidx, develmo

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

#def cosine(query_vec, question_vec):
#    num = dy.transpose(query_vec) * question_vec
#    dem1 = dy.sqrt(dy.transpose(query_vec) * query_vec)
#   dem2 = dy.sqrt(dy.transpose(question_vec) * question_vec)
#    dem = dem1 * dem2
#
#    return dy.cdiv(num, dem)

def frobenius_norm(query_emb, question_emb):
    query_emb = list(map(lambda x: dy.inputTensor(x), list(query_emb)))
    question_emb = list(map(lambda x: dy.inputTensor(x), list(question_emb)))

    frobenius = 0.0
    for i in range(len(query_emb)):
        for j in range(len(question_emb)):
            cos = dy.rectify(cosine(query_emb[i], question_emb[j])).value()
            frobenius += (cos**2)

    dy.renew_cg()
    return np.sqrt(frobenius)

class TreeKernel():
    def __init__(self, alpha=0, decay=1, ignore_leaves=True, smoothed=True):
        self.alpha = alpha
        self.decay = decay
        self.ignore_leaves = ignore_leaves
        self.smoothed = smoothed


    def __call__(self, query_tree, question_tree, query_emb=[], question_emb=[]):
        result = 0
        self.query_emb = query_emb
        self.question_emb = question_emb

        for node1 in query_tree['nodes']:
            node1_type = query_tree['nodes'][node1]['type']
            edgelen1 = len(query_tree['edges'][node1])
            for node2 in question_tree['nodes']:
                node2_type = question_tree['nodes'][node2]['type']
                if 'terminal' not in [node1_type, node2_type]:
                    edgelen2 = len(question_tree['edges'][node2])
                    delta = (self.decay**(edgelen1+edgelen2)) * self.__delta__(query_tree, question_tree, node1, node2)
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
                    result = cosine_similarity([self.query_emb[idx1]], [self.question_emb[idx2]])[0][0]
            else:
                result = 1
                for i in range(len(tree1['edges'][root1])):
                    if result == 0:
                        break
                    child1 = tree1['edges'][root1][i]
                    child2 = tree2['edges'][root2][i]
                    result *= (self.alpha + self.__delta__(tree1, tree2, child1, child2))
        return result


    def similar_terminals(self, query_tree, question_tree):
        for node1 in query_tree['nodes']:
            node1_type = query_tree['nodes'][node1]['type']
            for node2 in question_tree['nodes']:
                node2_type = question_tree['nodes'][node2]['type']

                if node1_type == 'terminal' and node2_type == 'terminal':
                    w1 = query_tree['nodes'][node1]['name'].replace('-rel', '').strip()
                    w2 = question_tree['nodes'][node2]['name'].replace('-rel', '').strip()

                    if w1 == w2:
                        if '-rel' not in query_tree['nodes'][node1]['name']:
                            query_tree['nodes'][node1]['name'] += '-rel'
                        if '-rel' not in question_tree['nodes'][node2]['name']:
                            question_tree['nodes'][node2]['name'] += '-rel'

                        # fathers
                        prev_id1 = query_tree['nodes'][node1]['parent']
                        if '-rel' not in query_tree['nodes'][prev_id1]['name']:
                            query_tree['nodes'][prev_id1]['name'] += '-rel'

                        prev_id2 = question_tree['nodes'][node2]['parent']
                        if '-rel' not in question_tree['nodes'][prev_id2]['name']:
                            question_tree['nodes'][prev_id2]['name'] += '-rel'

                        # grandfathers
                        prev_prev_id1 = query_tree['nodes'][prev_id1]['parent']
                        if '-rel' not in query_tree['nodes'][prev_prev_id1]['name']:
                            query_tree['nodes'][prev_prev_id1]['name'] += '-rel'

                        prev_prev_id2 = question_tree['nodes'][prev_id2]['parent']
                        if '-rel' not in question_tree['nodes'][prev_prev_id2]['name']:
                            question_tree['nodes'][prev_prev_id2]['name'] += '-rel'
        return query_tree, question_tree


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
