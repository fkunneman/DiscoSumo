__author__='thiagocastroferreira'

import _pickle as p
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import copy
import dynet as dy
import ev, metrics
import numpy as np
import os
import paths
import time
from sklearn.metrics import f1_score, accuracy_score
from semi import Semi

DEV_GOLD_PATH=paths.DEV_GOLD_PATH
EVALUATION_PATH='/roaming/tcastrof/semeval/acl2019'

def prepare_gold(path):
    ir = ev.read_res_file_aid(path, 'trec')
    return ir


def evaluate(ranking, gold):
    for qid in gold:
        gold_sorted = sorted(gold[qid], key = itemgetter(2), reverse = True)
        pred_sorted = ranking[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(2), reverse = True)

        gold[qid], ranking[qid] = [], []
        for i, row in enumerate(gold_sorted):
            relevant, gold_score, aid = row
            gold[qid].append((relevant, gold_score, aid))

            pred_score = pred_sorted[i][1]
            ranking[qid].append((relevant, pred_score, aid))

    for qid in gold:
        # Sort by IR score.
        gold_sorted = sorted(gold[qid], key = itemgetter(1), reverse = True)

        # Sort by SVM prediction score.
        pred_sorted = ranking[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(1), reverse = True)

        gold[qid] = [rel for rel, score, aid in gold_sorted]
        ranking[qid] = [rel for rel, score, aid in pred_sorted]

    map_gold = metrics.map(gold, 10)
    map_pred = metrics.map(ranking, 10)
    return map_gold, map_pred


def load_additional():
    path = os.path.join(paths.SEMI_PATH, 'training.pickle')

    if not os.path.exists(path):
        with open(os.path.join(paths.SEMI_PATH, 'index.txt')) as f:
            index = f.read().split('\n')

        with open(os.path.join(paths.SEMI_PATH, 'question.txt')) as f:
            questions = f.read().split('\n')
            questions = [snt.replace('<SENTENCE>', ' ').split() for snt in questions]

        idx2question = dict(zip(index, questions))
        additional = p.load(open(os.path.join(paths.SEMI_PATH, 'reranking'), 'rb'))
        for q1id in additional:
            q1 = idx2question[q1id]
            for q2id in additional[q1id]:
                q2 = idx2question[q2id]

                additional[q1id][q2id]['q1_full'] = ['EOS'] + q1[:80] + ['EOS']
                additional[q1id][q2id]['q2_full'] = ['EOS'] + q2[:80] + ['EOS']

        # select queries that have at least one negative question. Select the proportional number of positive examples
        new_additional = {}
        for q1id in additional:
            labels = [(q1id, q2id, additional[q1id][q2id]['label'], additional[q1id][q2id]['score']) for q2id in additional[q1id]]

            negative = [w for w in labels if w[2] == 0]
            if 0 < len(negative) <= 5:
                new_additional[q1id] = {}
                positive = sorted([w for w in labels if w[2] == 1], key=lambda x: x[-1], reverse=True)[:len(negative)]

                for row in negative:
                    _, q2id, label, score = row
                    new_additional[q1id][q2id] = additional[q1id][q2id]
                for row in positive:
                    _, q2id, label, score = row
                    new_additional[q1id][q2id] = additional[q1id][q2id]

        p.dump(new_additional, open(path, 'wb'))
    else:
        new_additional = p.load(open(path, 'rb'))

    vocab = ['UNK']
    questions = []
    negative, positive = 0, 0
    for q1id in new_additional:
        questions += [len(list(new_additional[q1id].keys()))]

        for q2id in new_additional[q1id]:
            vocab.extend(new_additional[q1id][q2id]['q1_full'])
            vocab.extend(new_additional[q1id][q2id]['q2_full'])

            if new_additional[q1id][q2id]['label'] == 0:
                negative += 1
            else:
                positive += 1

    vocab = list(set(vocab))
    voc2id = dict([(w, i) for i, w in enumerate(vocab)])

    mean = round(sum(questions) / float(len(questions)), 2)
    print('Training data -', 'Average number of pairs: ', mean, 'Positive examples: ', positive, 'Negative examples: ', negative, sep='\t')
    return new_additional, voc2id


def binarize(tree):
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


def parse_tree(tree):
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
                'name': terminal,
                'parent': prev_id,
                'type': 'terminal',
                'idx': terminalidx,
                'lemma': terminal.lower(),
            }

            terminalidx += 1
            edges[node_id] = []
            edges[prev_id].append(node_id)

        node_id += 1
        for i in range(len(closing)):
            prev_id = nodes[prev_id]['parent']
    return {'nodes': nodes, 'edges': edges, 'root': root}


def prepare_tree_vocabulary(indexset):
    def vocab(tree):
        nodes = tree['nodes']
        for node in nodes:
            type_ = nodes[node]['type']
            if type_ != 'terminal':
                name = nodes[node]['name']
                vocabulary.append(name)

    vocabulary = []

    for q1id in indexset:
        auxid = list(indexset[q1id].keys())[0]
        q1_tree = binarize(parse_tree(indexset[q1id][auxid]['q1_tree']))
        vocab(q1_tree)

        for q2id in indexset[q1id]:
            q2_tree = binarize(parse_tree(indexset[q1id][q2id]['q2_tree']))
            vocab(q2_tree)

    # UNKNOWN
    vocabulary.append('UNK')
    vocabulary = list(set(vocabulary))
    voc2id = [(w, i) for i, w in enumerate(vocabulary)]
    id2voc = [(w[1], w[0]) for w in voc2id]
    return dict(voc2id), dict(id2voc)


class SemevalTree(Semi):
    def __init__(self, properties):
        Semi.__init__(self, stop=False, lowercase=False, punctuation=False, w2v_dim=properties['EMB_DIM'])

        print('Load additional data...')
        self.additional, self.voc2id = load_additional()


        self.EPOCH = properties['EPOCH']
        self.BATCH = properties['BATCH']
        self.EMB_DIM = properties['EMB_DIM']
        self.HIDDEN_DIM = properties['HIDDEN_DIM']
        self.DROPOUT = properties['DROPOUT']
        self.EARLY_STOP = properties['EARLY_STOP']

        print('\nInitializing model...')
        print(self.fname())
        self.init()


    def init(self):
        dy.renew_cg()
        self.model = dy.Model()

        vocab = list(self.word2vec.wv.vocab)
        embeddings = []
        for w in vocab:
            embeddings.append(self.word2vec[w])

        vocab.append('UNK')
        embeddings.append(np.random.uniform(-0.1, 0.1, (self.w2v_dim)))
        self.voc2id = dict([(w, i) for i, w in enumerate(vocab)])
        self.terminal_lp = self.model.lookup_parameters_from_numpy(np.array(embeddings))

        self.tree2id, self.id2tree = prepare_tree_vocabulary(self.trainset)
        TREE_VOCAB_SIZE = len(self.tree2id)
        self.nonterminal_lp = self.model.add_lookup_parameters((TREE_VOCAB_SIZE, self.EMB_DIM))

        self.WS_query = [self.model.add_parameters((self.HIDDEN_DIM, self.EMB_DIM)) for _ in "iou"]
        self.US_query = [self.model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM + self.EMB_DIM)) for _ in "iou"]
        self.UFS_query = [self.model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM + self.EMB_DIM)) for _ in "ff"]
        self.BS_query = [self.model.add_parameters(self.HIDDEN_DIM) for _ in "iouf"]

        self.WS_question = [self.model.add_parameters((self.HIDDEN_DIM, self.EMB_DIM)) for _ in "iou"]
        self.US_question = [self.model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM + self.EMB_DIM)) for _ in "iou"]
        self.UFS_question = [self.model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM + self.EMB_DIM)) for _ in "ff"]
        self.BS_question = [self.model.add_parameters(self.HIDDEN_DIM) for _ in "iouf"]

        self.W = self.model.add_parameters((2, (self.HIDDEN_DIM*2)))
        self.bW = self.model.add_parameters((2))


    def expr_for_tree(self, root, tree, WS, US, UFS, BS):
        nodes, edges = tree['nodes'], tree['edges']
        if len(edges[root]) > 2: raise RuntimeError('Tree structure error: only binary trees are supported.')

        node_type = nodes[root]['type']
        if node_type == 'terminal': raise RuntimeError('Tree structure error: meet with leaves')

        if node_type == 'preterminal':
            terminal_id = edges[root][0]
            terminal = nodes[terminal_id]['name']

            try: idx = self.voc2id[terminal]
            except: idx = self.voc2id['UNK']

            emb = dy.lookup(self.terminal_lp, idx)
            Wi, Wo, Wu = [w for w in WS]
            bi, bo, bu, _ = [b for b in BS]
            i = dy.logistic(dy.affine_transform([bi, Wi, emb]))
            o = dy.logistic(dy.affine_transform([bo, Wo, emb]))
            u = dy.tanh(dy.affine_transform([bu, Wu, emb]))
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
        else:
            nonterminal = nodes[root]['name']
            try: idx = self.tree2id[nonterminal]
            except: idx = self.tree2id['UNK']
            emb = dy.lookup(self.nonterminal_lp, idx)

            e1, c1 = self.expr_for_tree(edges[root][0], tree, WS, US, UFS, BS)
            e2, c2 = self.expr_for_tree(edges[root][1], tree, WS, US, UFS, BS)
            Ui, Uo, Uu = [u for u in US]
            Uf1, Uf2 = [u for u in UFS]
            bi, bo, bu, bf = [b for b in BS]
            e = dy.concatenate([emb, e1, e2])
            i = dy.logistic(dy.affine_transform([bi, Ui, e]))
            o = dy.logistic(dy.affine_transform([bo, Uo, e]))
            f1 = dy.logistic(dy.affine_transform([bf, Uf1, e]))
            f2 = dy.logistic(dy.affine_transform([bf, Uf2, e]))
            u = dy.tanh(dy.affine_transform([bu, Uu, e]))
            c = dy.cmult(i, u) + dy.cmult(f1, c1) + dy.cmult(f2, c2)
            h = dy.cmult(o, dy.tanh(c))

        if self.DROPOUT > 0:
            dy.dropout(h, self.DROPOUT)
        return h, c


    def get_loss(self, q1, q2, label):
        q1_vec, _ = self.expr_for_tree(q1['root'], q1, self.WS_query, self.US_query, self.UFS_query, self.BS_query)
        q2_vec, _ = self.expr_for_tree(q2['root'], q2, self.WS_question, self.US_question, self.UFS_question, self.BS_question)

        x = dy.concatenate([q1_vec, q2_vec,])
        probs = dy.softmax(self.W * x + self.bW)
        loss = -dy.log(dy.pick(probs, label))
        return loss


    def load(self, path):
        self.model.populate(path)


    def fname(self):
        return '_'.join([str(self.EPOCH),
                         str(self.EMB_DIM),
                         str(self.HIDDEN_DIM),
                         str(self.EARLY_STOP),
                         str(self.BATCH),
                         str(self.DROPOUT)])


    def test(self, testset):
        ranking = {}
        y_real, y_pred = [], []

        for q1id in testset:
            auxid = list(testset[q1id].keys())[0]
            q1 = binarize(parse_tree(testset[q1id][auxid]['q1_tree']))
            q1_vec, _ = self.expr_for_tree(q1['root'], q1, self.WS_query, self.US_query, self.UFS_query, self.BS_query)

            for q2id in testset[q1id]:
                q2 = binarize(parse_tree(testset[q1id][q2id]['q2_tree']))
                q2_vec, _ = self.expr_for_tree(q2['root'], q2, self.WS_question, self.US_question, self.UFS_question, self.BS_question)


                x = dy.concatenate([q1_vec, q2_vec])
                probs = dy.softmax(self.W * x + self.bW).vec_value()
                score = dy.pick(probs, 1).value()

                probs = probs.vec_value()
                pred_label = probs.index(max(probs))
                y_pred.append(pred_label)

                label = testset[q1id][q2id]
                y_real.append(label)

                ranking[q1id].append((pred_label, score, q2id))
            dy.renew_cg()

        gold = prepare_gold(DEV_GOLD_PATH)
        map_baseline, map_model = evaluate(gold, ranking)
        f1score = f1_score(y_real, y_pred)
        accuracy = accuracy_score(y_real, y_pred)
        return map_baseline, map_model, f1score, accuracy


    def train(self, traindata, lr=1e-5):
        dy.renew_cg()
        trainer = dy.AdamTrainer(self.model, alpha=lr)

        early = 0.0
        best = -1
        for epoch in range(self.EPOCH):
            print('\n')
            dy.renew_cg()
            losses = []
            closs = 0
            batch_timing = []
            for i, trainrow in enumerate(traindata):
                start = time.time()
                q1 = binarize(parse_tree(trainrow['q1_tree']))
                q2 = binarize(parse_tree(trainrow['q2_tree']))
                label = trainrow['label']

                loss = self.get_loss(q1, q2, label)
                losses.append(loss)

                if len(losses) == self.BATCH:
                    loss = dy.esum(losses)
                    # loss += self.regularization_loss()
                    _loss = loss.value()
                    closs += _loss
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    # percentage of trainset processed
                    percentage = str(round((float(i+1) / len(traindata)) * 100,2)) + '%'
                    # time of epoch processing
                    time_epoch = round(sum(batch_timing) / float(len(batch_timing)), 2)

                    print("Epoch: {0} \t\t Loss: {1} \t\t Epoch time: {2} \t\t Trainset: {3}".format(epoch+1, round(_loss, 2), time_epoch, percentage), end='       \r')
                    losses = []
                    batch_timing = []
                end = time.time()
                t = (end-start)
                batch_timing.append(t)

            log = "Epoch: {0} \t\t Loss: {1} \t\t Best: {2}".format(epoch+1, round(closs/self.BATCH, 2), round(best, 2))
            print('\n' + log)

            log = 'Dev evaluation...'
            print(log)
            map_baseline, map_model, f1score, accuracy = self.test(self.devdata)

            results = 'MAP Model: {0} \t MAP baseline: {1} \t F1 score: {2} \t Accuracy: {3}'.format(round(map_model, 2), round(map_baseline, 2), round(f1score, 2), round(accuracy, 2))
            print(results)

            if map_model > best:
                best = copy.copy(map_model)
                early = 0
                # path = self.fname() + '.dy'
                # self.model.save(os.path.join(EVALUATION_PATH, path))
            else:
                trainer.learning_rate *= 0.5
                early += 1

            if early == self.EARLY_STOP:
                break

    def regularization_loss(self, coef=1e-4):
        losses = [dy.l2_norm(p) ** 2 for p in self.model.parameters_list()]
        return (coef / 2) * dy.esum(losses)

if __name__ == '__main__':
    properties = {
        'EPOCH': 30,
        'BATCH': 32,
        'EMB_DIM': 100,
        'HIDDEN_DIM': 128,
        'DROPOUT': 0.3,
        'EARLY_STOP': 5,
    }

    siamese = SemevalTree(properties)
    siamese.train(siamese.additional, lr=1e-4)
    siamese.train(siamese.traindata, lr=1e-5)