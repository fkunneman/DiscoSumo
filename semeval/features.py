__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/10/2018
Description:
    Script for extracting features for our ranking model
"""

import copy
import json
import nltk
import re
import time
from difflib import SequenceMatcher
from scipy.spatial import distance
from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

def lcsub(query, question):
    '''
    :param query:
    :param question:
    :return: longest common substring and size
    '''
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

    return float(len(query & question)) / len(query | question)

def containment_similarities(query, question, tokenize=False):
    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
    query = set(query.split())
    question = set(question.split())

    return float(len(query & question)) / len(query)
            
def greedy_string_tiling(query, question, tokenize=False):
    if tokenize:
        query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
        question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)
    query = query.split()
    question = question.split()

    if len(query) == 0 or len(text) == 0:
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

    return distance.dice(query, question)

class TreeKernel():
    def __init__(self, alpha=0):
        self.props={'annotators': 'tokenize,ssplit,pos,parse','pipelineLanguage':'en','outputFormat':'json'}
        self.corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')
        self.alpha = alpha

    def __call__(self, query, question, tokenize=False):
        if tokenize:
            query = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)
            question = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', question)

        query_tree = self.constituency_tree(query)
        question_tree = self.constituency_tree(question)

        return self.tree_kernel(query_tree, question_tree)


    def parse_tree(self, tree):
        nodes, edges, root = {}, {}, 1
        node_id = 1
        prev_id = 0

        for child in tree.replace('\n', '').split():
            closing = list(filter(lambda x: x == ')', child))
            if child[0] == '(':
                nodes[node_id] = {
                    'id': node_id,
                    'name': child[1:],
                    'parent': prev_id,
                    'type': 'nonterminal'
                }
                edges[node_id] = []

                if prev_id > 0:
                    edges[prev_id].append(node_id)
                prev_id = copy.copy(node_id)
            else:
                terminal = child.replace(')', '')
                nodes[prev_id]['type'] = 'preterminal'

                nodes[node_id] = {
                    'id': node_id,
                    'name': terminal.lower(),
                    'parent': prev_id,
                    'type': 'terminal'
                }
                edges[node_id] = []
                edges[prev_id].append(node_id)

            node_id += 1
            for i in range(len(closing)):
                prev_id = nodes[prev_id]['parent']
        return {'nodes': nodes, 'edges': edges, 'root': root}


    def constituency_tree(self, question):
        out = json.loads(self.corenlp.annotate(question, properties=self.props))

        sentences = '(SENTENCES '
        for snt in out['sentences']:
            sentences += snt['parse'].replace('\n', '') + ' '
        sentences = sentences.strip()
        sentences += ')'

        tree = self.parse_tree(sentences)
        return tree


    def __is_same_production__(self, tree1, tree2, root1, root2):
        production1 = tree1['nodes'][root1]['name'] + ' -> '
        for child in tree1['edges'][root1]:
            production1 += tree1['nodes'][child]['name'] + ' '

        production2 = tree1['nodes'][root2]['name'] + ' -> '
        for child in tree1['edges'][root2]:
            production2 += tree2['nodes'][child]['name'] + ' '

        if production1.strip() == production2.strip():
            return True
        else:
            return False


    def __delta__(self, tree1, tree2, root1, root2):
        if self.__is_same_production__(tree1, tree2, root1, root2):
            node1_type = tree1['nodes'][root1]['type']
            node2_type = tree2['nodes'][root2]['type']
            if node1_type == 'preterminal' and node2_type == 'preterminal':
                return 1
            else:
                result = 1
                for i in range(len(tree1['edges'])):
                    if result == 0:
                        break
                    child1 = tree1['edges'][i]
                    child2 = tree2['edges'][i]
                    result *= self.__delta__(tree1, tree2, child1, child2)
                return result
        return 0


    def tree_kernel(self, tree1, tree2):
        result = 0
        for node1 in tree1['nodes']:
            for node2 in tree2['nodes']:
                node1_type = tree1['nodes'][node1]['type']
                node2_type = tree2['nodes'][node2]['type']
                if 'terminal' not in [node1_type, node2_type]:
                    delta = self.__delta__(tree1, tree2, node1, node2)
                    result += delta
                    if delta == 1:
                        break

        return result


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
