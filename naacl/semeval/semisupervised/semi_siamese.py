__author__='thiagocastroferreira'

import _pickle as p
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('/roaming/tcastrof/semeval/evaluation/MAP_scripts')
import copy
import dynet as dy
import ev, metrics
import os
import numpy as np
import time
import paths
from sklearn.metrics import f1_score
from operator import itemgetter
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


class SemiSiamese(Semi):
    def __init__(self, properties):
        Semi.__init__(self, stop=False, lowercase=False, punctuation=False, w2v_dim=properties['EMB_DIM'])

        print('Load additional data...')
        self.additional, self.voc2id = load_additional()

        self.MODEL = properties['MODEL']
        self.EPOCH = properties['EPOCH']
        self.BATCH = properties['BATCH']
        self.EMB_DIM = properties['EMB_DIM']
        self.HIDDEN_DIM = properties['HIDDEN_DIM']
        self.DROPOUT = properties['DROPOUT']
        self.EARLY_STOP = properties['EARLY_STOP']

        print('\nInitializing model...')
        self.init()


    def init(self):
        dy.renew_cg()
        self.model = dy.Model()

        # VOCAB_SIZE = len(self.voc2id)
        # self.lp = self.model.add_lookup_parameters((VOCAB_SIZE, self.EMB_DIM))

        vocab = list(self.word2vec.wv.vocab)
        embeddings = []
        for w in vocab:
            embeddings.append(self.word2vec[w])

        vocab.append('UNK')
        embeddings.append(np.random.uniform(-0.1, 0.1, (300)))
        self.voc2id = dict([(w, i) for i, w in enumerate(vocab)])
        self.lp = self.model.lookup_parameters_from_numpy(np.array(embeddings))

        if self.MODEL == 'conv':
            self.init_conv()
        elif self.MODEL == 'lstm':
            self.init_lstm()

        self.W = self.model.add_parameters((2, (self.HIDDEN_DIM*2)))
        self.bW = self.model.add_parameters((2))


    def init_conv(self):
        # QUERY
        self.F_query = self.model.add_parameters((5, self.EMB_DIM, 1, 100))
        self.b_query = self.model.add_parameters((100, ))

        # CANDIDATE QUESTION
        self.F_question = self.model.add_parameters((5, self.EMB_DIM, 1, 100))
        self.b_question = self.model.add_parameters((100, ))

        input_size = 1 * (self.EMB_DIM * 100)
        self.W1_query = self.model.add_parameters((self.HIDDEN_DIM, input_size))
        self.bW1_query = self.model.add_parameters((self.HIDDEN_DIM))
        dy.dropout(self.W1_query, self.DROPOUT)

        self.W1_question = self.model.add_parameters((self.HIDDEN_DIM, input_size))
        self.bW1_question = self.model.add_parameters((self.HIDDEN_DIM))
        dy.dropout(self.W1_question, self.DROPOUT)


    def init_lstm(self):
        self.fwd_lstm_query = dy.LSTMBuilder(1, self.EMB_DIM, 512, self.model)
        self.bwd_lstm_query = dy.LSTMBuilder(1, self.EMB_DIM, 512, self.model)

        self.fwd_lstm_query.set_dropout(self.DROPOUT)
        self.bwd_lstm_query.set_dropout(self.DROPOUT)

        self.fwd_lstm_question = dy.LSTMBuilder(1, self.EMB_DIM, 512, self.model)
        self.bwd_lstm_question = dy.LSTMBuilder(1, self.EMB_DIM, 512, self.model)

        self.fwd_lstm_question.set_dropout(self.DROPOUT)
        self.bwd_lstm_question.set_dropout(self.DROPOUT)

        input_size = 2 * 512
        self.W1_query = self.model.add_parameters((self.HIDDEN_DIM, input_size))
        self.bW1_query = self.model.add_parameters((self.HIDDEN_DIM))

        self.W1_question = self.model.add_parameters((self.HIDDEN_DIM, input_size))
        self.bW1_question = self.model.add_parameters((self.HIDDEN_DIM))


    def __embed__(self, text):
        question = []
        index = []
        for w in text:
            question.append(w)
            try:
                _id = self.voc2id[w]
            except:
                _id = self.voc2id['UNK']
            index.append(_id)

        embeddings = list(map(lambda idx: self.lp[idx], index))
        return embeddings


    def __convolve__(self, embeddings, F, b, W1, bW1):
        sntlen = len(embeddings)
        emb = dy.concatenate_cols(embeddings)

        x = dy.conv2d_bias(emb, F, b, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        x = dy.rectify(x)
        if self.DROPOUT > 0:
            dy.dropout(x, self.DROPOUT)
        f = dy.reshape(x, (self.EMB_DIM * 1 * 100,))

        return W1 * f + bW1


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def __recur__(self, embeddings, fwd_lstm, bwd_lstm, W1, bW1):
        embeddings_rev = list(reversed(embeddings))

        # lpidx = self.lp[self.voc2id['eos']]
        fwd_vectors = self.run_lstm(fwd_lstm.initial_state(), embeddings)
        bwd_vectors = self.run_lstm(bwd_lstm.initial_state(), embeddings_rev)
        bwd_vectors = list(reversed(bwd_vectors))

        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        x = dy.average(vectors)
        return W1 * x + bW1


    def forward(self, query, question):
        query_embedding = self.__embed__(query)

        query_vec = None
        if self.MODEL == 'conv':
            query_vec = self.__convolve__(query_embedding,
                                          self.F_query,
                                          self.b_query,
                                          self.W1_query,
                                          self.bW1_query)
        elif self.MODEL == 'lstm':
            query_vec = self.__recur__(query_embedding,
                                       self.fwd_lstm_query,
                                       self.bwd_lstm_query,
                                       self.W1_query,
                                       self.bW1_query)

        question_embedding = self.__embed__(question)

        question_vec = None
        if self.MODEL == 'conv':
            question_vec = self.__convolve__(question_embedding,
                                             self.F_question,
                                             self.b_question,
                                             self.W1_question,
                                             self.bW1_question)
        elif self.MODEL == 'lstm':
            question_vec = self.__recur__(question_embedding,
                                          self.fwd_lstm_question,
                                          self.bwd_lstm_question,
                                          self.W1_question,
                                          self.bW1_question)

        return query_vec, question_vec


    def get_loss(self, query, question, label):
        # forward
        query_vec, question_vec = self.forward(query, question)

        x = dy.concatenate([query_vec, question_vec])
        probs = dy.softmax(self.W * x + self.bW)
        loss = -dy.log(dy.pick(probs, label))

        return loss


    def load(self, path):
        self.model.populate(path)


    def fname(self):
        return '_'.join([str(self.MODEL),
                         str(self.EPOCH),
                         str(self.EMB_DIM),
                         str(self.HIDDEN_DIM),
                         str(self.EARLY_STOP),
                         str(self.BATCH),
                         str(self.DROPOUT)])


    def tepoch(self, epoch_timing):
        time_epoch = sum(epoch_timing)
        if time_epoch > 3600:
            time_epoch = str(round(time_epoch / 3600, 2)) + ' h'
        elif time_epoch > 60:
            time_epoch = str(round(time_epoch / 60, 2)) + ' min'
        else:
            time_epoch = str(round(time_epoch, 2)) + ' sec'
        return time_epoch


    def train(self, traindata, lr=1e-5):
        dy.renew_cg()
        trainer = dy.AdamTrainer(self.model, alpha=lr)

        # Loggin
        path = self.fname() + '.log' if len(traindata) > 5000 else self.fname() + 'tuning.log'
        f = open(os.path.join(EVALUATION_PATH, path), 'w')

        epoch_timing = []
        early = 0.0
        best = -1
        for epoch in range(self.EPOCH):
            print('\n')
            dy.renew_cg()
            losses = []
            closs = 0
            batch_timing = []
            for i, q1id in enumerate(traindata):
                for q2id in traindata[q1id]:
                    start = time.time()
                    query = traindata[q1id][q2id]['q1_full']
                    question = traindata[q1id][q2id]['q2_full']
                    label = traindata[q1id][q2id]['label']

                    loss = self.get_loss(query, question, label)
                    losses.append(loss)

                    if len(losses) == self.BATCH:
                        loss = dy.esum(losses)
                        _loss = loss.value()
                        closs += _loss
                        loss.backward()
                        trainer.update()
                        dy.renew_cg()

                        # percentage of trainset processed
                        percentage = str(round((float(i+1) / len(traindata)) * 100,2)) + '%'
                        # time of epoch processing
                        time_epoch = self.tepoch(epoch_timing)
                        print("Epoch: {0} \t\t Loss: {1} \t\t Epoch time: {2} \t\t Trainset: {3}".format(epoch+1, round(_loss, 2), time_epoch, percentage), end='       \r')
                        losses = []
                        batch_timing = []
                    end = time.time()
                    t = (end-start)
                    batch_timing.append(t)
                    epoch_timing.append(t)

            log = "Epoch: {0} \t\t Loss: {1} \t\t Best: {2}".format(epoch+1, round(closs/self.BATCH, 2), round(best, 2))
            print('\n' + log)
            f.write(' '.join([log, '\n']))

            log = 'Dev evaluation...'
            print(log)
            f.write(log + '\n')
            map_baseline, map_model, f1score = self.test(self.devdata)

            print('MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), 'F1 score: ', str(round(f1score, 2)), sep='\t', end='\n')
            f.write('\t'.join(['MAP Model: ', str(round(map_model, 2)), 'MAP baseline: ', str(round(map_baseline, 2)), 'F1 score: ', str(round(f1score, 2)), '\n']))

            epoch_timing = []
            if map_model > best:
                best = copy.copy(map_model)
                early = 0
                path = self.fname() + '.dy'
                self.model.save(os.path.join(EVALUATION_PATH, path))
            else:
                early += 1

            if early == self.EARLY_STOP:
                break
        f.close()


    def test(self, testset):
        ranking = {}
        y_real, y_pred = [], []
        for i, q1id in enumerate(testset):
            ranking[q1id] = []
            percentage = round(float(i+1) / len(testset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')

            q2id = list(testset[q1id].keys())[0]
            q1 = ['EOS'] + testset[q1id][q2id]['q1_full'] + ['EOS']
            query_embedding = self.__embed__(q1)

            query_vec = None
            if self.MODEL == 'conv':
                query_vec = self.__convolve__(query_embedding,
                                              self.F_query,
                                              self.b_query,
                                              self.W1_query,
                                              self.bW1_query)
            elif self.MODEL == 'lstm':
                query_vec = self.__recur__(query_embedding,
                                           self.fwd_lstm_query,
                                           self.bwd_lstm_query,
                                           self.W1_query,
                                           self.bW1_query)

            for q2id in testset[q1id]:
                q2 = ['EOS'] + testset[q1id][q2id]['q2_full'] + ['EOS']

                question_embedding = self.__embed__(q2)

                question_vec = None
                if self.MODEL == 'conv':
                    question_vec = self.__convolve__(question_embedding,
                                                     self.F_question,
                                                     self.b_query,
                                                     self.W1_question,
                                                     self.bW1_question)
                elif self.MODEL == 'lstm':
                    question_vec = self.__recur__(question_embedding,
                                                  self.fwd_lstm_question,
                                                  self.bwd_lstm_question,
                                                  self.W1_question,
                                                  self.bW1_question)

                x = dy.concatenate([query_vec, question_vec])
                probs = dy.softmax(self.W * x + self.bW)
                score = dy.pick(probs, 1).value()

                probs = probs.vec_value()
                pred_label = probs.index(max(probs))
                y_pred.append(pred_label)

                ranking[q1id].append((pred_label, score, q2id))
                y_real.append(testset[q1id][q2id]['label'])
            dy.renew_cg()

        map_baseline, map_model = evaluate(copy.copy(ranking), prepare_gold(DEV_GOLD_PATH))
        f1score = f1_score(y_real, y_pred)
        return map_baseline, map_model, f1score

if __name__ == '__main__':
    if not os.path.exists(EVALUATION_PATH):
        os.mkdir(EVALUATION_PATH)

    # CONV
    properties = {
        'EPOCH': 30,
        'BATCH': 50,
        'EMB_DIM': 100,
        'HIDDEN_DIM': 128,
        'DROPOUT': 0.0,
        'EARLY_STOP': 5,
        'MODEL': 'conv',
    }

    siamese = SemiSiamese(properties)
    siamese.train(siamese.additional, lr=1e-4)
    siamese.train(siamese.traindata, lr=1e-5)

    ## LSTMs
    properties = {
        'EPOCH': 30,
        'BATCH': 32,
        'EMB_DIM': 100,
        'HIDDEN_DIM': 128,
        'DROPOUT': 0.2,
        'EARLY_STOP': 5,
        'MODEL': 'lstm'
    }

    siamese = SemiSiamese(properties)
    siamese.train(siamese.additional, lr=1e-4)
    siamese.train(siamese.traindata, lr=1e-5)