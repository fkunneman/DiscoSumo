__author__='thiagocastroferreira'

import copy
from sklearn.metrics.pairwise import cosine_similarity

class TreeKernel():
    def __init__(self, alpha=0, decay=1, ignore_leaves=True, smoothed=True):
        self.alpha = alpha
        self.decay = decay
        self.ignore_leaves = ignore_leaves
        self.smoothed = smoothed


    def __call__(self, q1_tree, q1_emb, q1_token2lemma, q2_tree, q2_emb, q2_token2lemma):
        result = 0
        self.q1_emb = q1_emb
        self.q2_emb = q2_emb

        q1_tree = self.binarize(self.parse_tree(q1_tree, q1_token2lemma))
        q2_tree = self.binarize(self.parse_tree(q2_tree, q2_token2lemma))


        q1_tree, q2_tree = self.similar_terminals(q1_tree, q2_tree)
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


    def __delta__(self, q1_tree, q2_tree, q1_root, q2_root):
        if 'production' not in q1_tree['nodes'][q1_root]:
            q1_tree['nodes'][q1_root]['production'] = self.get_production(q1_tree, q1_root)
        if 'production' not in q2_tree['nodes'][q2_root]:
            q2_tree['nodes'][q2_root]['production'] = self.get_production(q2_tree, q2_root)

        production1 = q1_tree['nodes'][q1_root]['production']
        production2 = q2_tree['nodes'][q2_root]['production']
        result = 0
        if production1 == production2:
            node1_type = q1_tree['nodes'][q1_root]['type']
            node2_type = q2_tree['nodes'][q2_root]['type']
            if node1_type == 'preterminal' and node2_type == 'preterminal':
                if not self.smoothed:
                    result = 1
                else:
                    child1 = q1_tree['edges'][q1_root][0]
                    child2 = q2_tree['edges'][q2_root][0]

                    idx1 = q1_tree['nodes'][child1]['idx']
                    idx2 = q2_tree['nodes'][child2]['idx']
                    # result = cosine_similarity([self.q1_emb[idx1]], [self.q2_emb[idx2]])[0][0]
                    result = max(0, cosine_similarity([self.q1_emb[idx1]], [self.q2_emb[idx2]])[0][0])**2
            else:
                result = 1
                for i in range(len(q1_tree['edges'][q1_root])):
                    if result == 0:
                        break
                    child1 = q1_tree['edges'][q1_root][i]
                    child2 = q2_tree['edges'][q2_root][i]
                    result *= (self.alpha + self.__delta__(q1_tree, q2_tree, child1, child2))
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


    # utilities
    def print_tree(self, root, tree, stree=''):
        if tree['nodes'][root]['type'] == 'terminal':
            stree += ' ' + tree['nodes'][root]['name']
        else:
            stree += '(' + tree['nodes'][root]['name'] + ' '

        for node in tree['edges'][root]:
            stree = self.print_tree(node, tree, stree) + ')'
        return stree


    def binarize(self, tree):
        # get nodes and edges of the tree
        nodes, edges = tree['nodes'], tree['edges']
        # get new possible id
        new_id = len(nodes)+1

        finished = False
        # remove nonterminal nodes with one child
        while not finished:
            finished = True
            # iterate over the tree nodes
            for root in nodes:
                if nodes[root]['type'] == 'nonterminal' and len(edges[root]) == 1:
                    child = edges[root][0]
                    parent = nodes[root]['parent']

                    # update who is the parent of the child
                    nodes[child]['parent'] = parent
                    # update the new child of the parent
                    if parent > 0:
                        for i, edge in enumerate(edges[parent]):
                            if edge == root:
                                edges[parent][i] = child
                                break
                    # delete root
                    del nodes[root]
                    del edges[root]
                    finished = False
                    break

        finished = False
        while not finished:
            finished = True
            # iterate over the tree nodes
            for root in nodes:
                # if root has mode than two children
                if len(edges[root]) > 2:
                    # root tag
                    name = nodes[root]['name']
                    # first children tag
                    child_name = nodes[edges[root][0]]['name']
                    # create new node
                    nodes[new_id] = {
                        'id': new_id,
                        'name': ''.join(['@', name, '_', child_name]),
                        'parent': root,
                        'type': 'nonterminal'
                    }
                    # new node assumes all children of root except first
                    edges[new_id] = edges[root][1:]
                    for child in edges[new_id]:
                        nodes[child]['parent'] = new_id
                    # new node becomes a child of root
                    edges[root] = [edges[root][0], new_id]

                    new_id += 1
                    finished = False
                    break

        tree['root'] = min(list(nodes.keys()))
        return tree


    def parse_tree(self, tree, token2lemma={}):
        nodes, edges, root = {}, {}, 1
        node_id = 1
        prev_id = 0
        terminalidx = 0

        for child in tree.replace('\n', '').split():
            closing = list(filter(lambda x: x == ')', child))
            if child[0] == '(':
                nodes[node_id] = {
                    'id': node_id,
                    'name': child[1:],
                    'parent': prev_id,
                    'type': 'nonterminal',
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
                    'type': 'terminal',
                    'idx': terminalidx,
                    'lemma': terminal.lower(),
                }
                if terminal.lower() in token2lemma:
                    nodes[node_id]['lemma'] = token2lemma[terminal.lower()].lower()

                terminalidx += 1
                edges[node_id] = []
                edges[prev_id].append(node_id)

            node_id += 1
            for i in range(len(closing)):
                prev_id = nodes[prev_id]['parent']
        return {'nodes': nodes, 'edges': edges, 'root': root}