__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/10/2018
Description:
    Script for extracting features for our ranking model
"""

import copy
import json
import re
import time
from scipy.spatial import distance
from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

def lcs(query, question):
    '''
    Longest common subsequences
    Reference: https://bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php
    :param query: query question + query body
    :param question: question + body
    :return:
    '''
    m = len(query)
    n = len(question)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if query[i] == question[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(query[i - c + 1:i + 1])
                elif c == longest:
                    lcs_set.add(query[i - c + 1:i + 1])

    return len(list(lcs_set)[0]), lcs_set

def jaccard(query, question, tokenize=False):
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

    return (float(len(query & question)) / len(query)
 
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
        # nlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')
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
    question = 'a cat eats a mouse.'
    query = 'a mouse eats a cat.'

    start = time.time()
    print(lcs(query, question))
    end = time.time()
    print(end-start)
    print(10 * '-')
    start = time.time()
    print(jaccard(query, question, True))
    end = time.time()
    print(end-start)
    print(10 * '-')
    start = time.time()
    print(dice(query, question, True))
    end = time.time()
    print(end-start)
