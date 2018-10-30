__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import copy
import dynet as dy
import ev, metrics
import features
import json
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import re
import time
import utils
from sklearn.metrics import f1_score

import preprocessing
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

EVALUATION_PATH='siamese'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

DATA_PATH='data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class SemevalTree():
    def __init__(self, properties):
        if not os.path.exists(DEV_PATH):
            preprocessing.run()

        logging.info('Preparing development set...')
        self.devset = json.load(open(DEV_PATH))
        self.devdata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.devset)

        logging.info('Preparing trainset...')
        self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.trainset)
        info = 'TRAIN DATA SIZE: ' + str(len(self.traindata))
        logging.info(info)


        self.EPOCH = properties['EPOCH']
        self.BATCH = properties['BATCH']
        self.EMB_DIM = properties['EMB_DIM']
        self.HIDDEN_DIM = properties['HIDDEN_DIM']
        self.DROPOUT = properties['DROPOUT']
        self.EARLY_STOP = properties['EARLY_STOP']
        self.pretrained = properties['pretrained_input']

        print('\nInitializing model...')
        print(self.fname())
        self.init()


    def init(self):
        dy.renew_cg()
        self.model = dy.Model()

        if self.pretrained:
            embeddings, self.voc2id, self.id2voc = features.init_glove()
            self.terminal_lp = self.model.lookup_parameters_from_numpy(embeddings)
        else:
            VOCAB_SIZE = len(self.voc2id)
            self.terminal_lp = self.model.add_lookup_parameters((VOCAB_SIZE, self.EMB_DIM))

        self.tree2id, self.id2tree = utils.prepare_tree_vocabulary(self.trainset)
        TREE_VOCAB_SIZE = len(self.tree2id)
        self.nonterminal_lp = self.model.add_lookup_parameters((TREE_VOCAB_SIZE, self.EMB_DIM))

        self.WS = [self.model.add_parameters((self.HIDDEN_DIM, self.EMB_DIM)) for _ in "iou"]
        self.US = [self.model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM + self.EMB_DIM)) for _ in "iou"]
        self.UFS = [self.model.add_parameters((self.HIDDEN_DIM, 2 * self.HIDDEN_DIM + self.EMB_DIM)) for _ in "ff"]
        self.BS = [self.model.add_parameters(self.HIDDEN_DIM) for _ in "iouf"]

        self.W = self.model.add_parameters((2, (self.HIDDEN_DIM*2)))
        self.bW = self.model.add_parameters((2))


    def expr_for_tree(self, root, tree):
        nodes, edges = tree['nodes'], tree['edges']
        if len(edges[root]) > 2: raise RuntimeError('Tree structure error: only binary trees are supported.')

        node_type = nodes[root]['type']
        if node_type == 'terminal': raise RuntimeError('Tree structure error: meet with leaves')

        if node_type == 'preterminal':
            terminal_id = edges[root][0]
            terminal = nodes[terminal_id]['name']

            try: idx = self.voc2id[terminal]
            except: idx = self.voc2id['UNK']

            emb = dy.lookup(self.terminal_lp, idx, update=False)
            Wi, Wo, Wu = [w for w in self.WS]
            bi, bo, bu, _ = [b for b in self.BS]
            i = dy.logistic(dy.affine_transform([bi, Wi, emb]))
            o = dy.logistic(dy.affine_transform([bo, Wo, emb]))
            u = dy.tanh(dy.affine_transform([bu, Wu, emb]))
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
            return h, c


        nonterminal = nodes[root]['name']
        try: idx = self.tree2id[nonterminal]
        except: idx = self.tree2id['UNK']
        emb = dy.lookup(self.nonterminal_lp, idx)

        e1, c1 = self.expr_for_tree(edges[root][0], tree)
        e2, c2 = self.expr_for_tree(edges[root][1], tree)
        Ui, Uo, Uu = [u for u in self.US]
        Uf1, Uf2 = [u for u in self.UFS]
        bi, bo, bu, bf = [b for b in self.BS]
        e = dy.concatenate([emb, e1, e2])
        i = dy.logistic(dy.affine_transform([bi, Ui, e]))
        o = dy.logistic(dy.affine_transform([bo, Uo, e]))
        f1 = dy.logistic(dy.affine_transform([bf, Uf1, e]))
        f2 = dy.logistic(dy.affine_transform([bf, Uf2, e]))
        u = dy.tanh(dy.affine_transform([bu, Uu, e]))
        c = dy.cmult(i, u) + dy.cmult(f1, c1) + dy.cmult(f2, c2)
        h = dy.cmult(o, dy.tanh(c))
        return h, c


    def get_loss(self, q1, q2, label):
        q1_vec, _ = self.expr_for_tree(q1['root'], q1)
        q2_vec, _ = self.expr_for_tree(q2['root'], q2)

        if self.DROPOUT > 0:
            dy.dropout(q1_vec, self.DROPOUT)
            dy.dropout(q2_vec, self.DROPOUT)

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
                         str(self.DROPOUT),
                         str(self.pretrained)])


    def test(self, testset):
        ranking = {}
        y_real, y_pred = [], []
        for i, qid in enumerate(testset):
            ranking[qid] = []
            percentage = round(float(i+1) / len(testset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')

            q1 = testset[qid]
            q1 = utils.binarize(utils.parse_tree(q1['tree']))
            q1_vec, _ = self.expr_for_tree(q1['root'], q1)


            duplicates = testset[qid]['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                rel_question_id = rel_question['id']
                q2 = rel_question['tree']
                q2 = utils.binarize(utils.parse_tree(q2))
                q2_vec, _ = self.expr_for_tree(q2['root'], q2)

                x = dy.concatenate([q1_vec, q2_vec])
                probs = dy.softmax(self.W * x + self.bW).vec_value()
                score = probs.index(max(probs))
                y_pred.append(score)

                if rel_question['relevance'] != 'Irrelevant':
                    y_real.append(1)
                else:
                    y_real.append(0)
                ranking[qid].append((score, score, rel_question_id))
            dy.renew_cg()

        gold = utils.prepare_gold(GOLD_PATH)
        map_baseline, map_model = utils.evaluate(gold, ranking)
        f1score = f1_score(y_real, y_pred)
        return map_baseline, map_model, f1score


    def train(self):
        dy.renew_cg()
        trainer = dy.AdamTrainer(self.model)

        early = 0.0
        best = -1
        for epoch in range(self.EPOCH):
            print('\n')
            dy.renew_cg()
            losses = []
            closs = 0
            batch_timing = []
            for i, trainrow in enumerate(self.traindata):
                start = time.time()
                q1 = utils.binarize(utils.parse_tree(trainrow['q1_tree']))
                q2 = utils.binarize(utils.parse_tree(trainrow['q2_tree']))
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
                    percentage = str(round((float(i+1) / len(self.traindata)) * 100,2)) + '%'
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
            map_baseline, map_model, f1score = self.test(self.devset)

            print('MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), 'F1 score: ', str(round(f1score, 2)), sep='\t', end='\n')

            trainer.learning_rate *= 0.95
            if map_model > best:
                best = copy.copy(map_model)
                early = 0
                # path = self.fname() + '.dy'
                # self.model.save(os.path.join(EVALUATION_PATH, path))
            else:
                early += 1

            if early == self.EARLY_STOP:
                break

    def regularization_loss(self, coef=1e-4):
        losses = [dy.l2_norm(p) ** 2 for p in self.model.parameters_list()]
        return (coef / 2) * dy.esum(losses)

if __name__ == '__main__':
    properties = {
        'EPOCH': 30,
        'BATCH': 8,
        'EMB_DIM': 300,
        'HIDDEN_DIM': 512,
        'DROPOUT': 0.3,
        'EARLY_STOP': 10,
        'pretrained_input': True
    }

    siamese = SemevalTree(properties)
    siamese.train()