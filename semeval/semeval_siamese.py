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

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

EVALUATION_PATH='siamese'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

GLOVE_PATH='/home/tcastrof/workspace/glove/glove.6B.300d.txt'

DATA_PATH='data'
TRAIN_PATH=os.path.join(DATA_PATH, 'trainset.data')
DEV_PATH=os.path.join(DATA_PATH, 'devset.data')

class SemevalSiamese():
    def __init__(self, properties, trainset, devset, testset):
        props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
        corenlp = StanfordCoreNLP(r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27')

        self.INPUT = properties['INPUT']
        self.MODEL = properties['MODEL']
        self.EPOCH = properties['EPOCH']
        self.BATCH = properties['BATCH']
        self.EMB_DIM = properties['EMB_DIM']
        self.HIDDEN_DIM = properties['HIDDEN_DIM']
        self.ERROR = properties['ERROR']
        self.DROPOUT = properties['DROPOUT']
        self.EARLY_STOP = properties['EARLY_STOP']
        self.m = properties['m']
        self.pretrained = properties['pretrained_input']

        print('Preparing trainset...')
        if not os.path.exists(TRAIN_PATH):
            self.trainset = utils.prepare_corpus(trainset, corenlp=corenlp, props=props)
            json.dump(self.trainset, open(TRAIN_PATH, 'w'))
        else:
            self.trainset = json.load(open(TRAIN_PATH))
        self.traindata, self.voc2id, self.id2voc, self.vocabulary = utils.prepare_traindata(self.trainset)
        print('TRAIN DATA SIZE: ', len(self.traindata))
        print('\nPreparing development set...')
        if not os.path.exists(DEV_PATH):
            self.devset = utils.prepare_corpus(devset, corenlp=corenlp, props=props)
            json.dump(self.devset, open(DEV_PATH, 'w'))
        else:
            self.devset = json.load(open(DEV_PATH))
        self.devgold = utils.prepare_gold(GOLD_PATH)
        print('\nPreparing test set...')
        self.testset = []

        corenlp.close()

        self.translation = features.init_translation(traindata=self.traindata,
                                                     vocabulary=self.vocabulary,
                                                     alpha=0.6,
                                                     sigma=0.6)

        self.trainidx, self.trainelmo, self.devidx, self.develmo = features.init_elmo()

        print('\nInitializing model...')
        self.FEATURE_DIM = len(self.__transform__(self.traindata[0]['q1'], self.traindata[0]['q2']))
        print(self.fname())
        self.init()


    def init(self):
        dy.renew_cg()
        self.model = dy.Model()

        if self.pretrained:
            embeddings, self.voc2id, self.id2voc = features.init_glove()
            self.lp = self.model.lookup_parameters_from_numpy(embeddings)
        else:
            VOCAB_SIZE = len(self.voc2id)
            self.lp = self.model.add_lookup_parameters((VOCAB_SIZE, self.EMB_DIM))

        if self.MODEL[:4] == 'conv':
            self.init_conv()
        elif self.MODEL == 'lstm':
            self.init_lstm()

        if self.ERROR == 'entropy':
            self.W = self.model.add_parameters((2, (self.HIDDEN_DIM*2)+self.FEATURE_DIM))
            self.bW = self.model.add_parameters((2))


    def init_conv(self):
        # QUERY
        # 4 filters bi- tri- four- and five-grams with 2 feature maps
        # self.F1_query = self.model.add_parameters((2, self.EMB_DIM, 1, 2))
        # self.b1_query = self.model.add_parameters((2, ))
        # self.F2_query = self.model.add_parameters((3, self.EMB_DIM, 1, 2))
        # self.b2_query = self.model.add_parameters((2, ))
        # self.F3_query = self.model.add_parameters((4, self.EMB_DIM, 1, 2))
        # self.b3_query = self.model.add_parameters((2, ))
        self.F4_query = self.model.add_parameters((5, self.EMB_DIM, 1, 2))
        self.b4_query = self.model.add_parameters((2, ))

        # dy.dropout(self.F1_query, self.DROPOUT)
        # dy.dropout(self.b1_query, self.DROPOUT)
        # dy.dropout(self.F2_query, self.DROPOUT)
        # dy.dropout(self.b2_query, self.DROPOUT)
        # dy.dropout(self.F3_query, self.DROPOUT)
        # dy.dropout(self.b3_query, self.DROPOUT)
        dy.dropout(self.F4_query, self.DROPOUT)
        dy.dropout(self.b4_query, self.DROPOUT)

        # CANDIDATE QUESTION
        # 4 filters bi- tri- four- and five-grams with 2 feature maps
        # self.F1_question = self.model.add_parameters((2, self.EMB_DIM, 1, 2))
        # self.b1_question = self.model.add_parameters((2, ))
        # self.F2_question = self.model.add_parameters((3, self.EMB_DIM, 1, 2))
        # self.b2_question = self.model.add_parameters((2, ))
        # self.F3_question = self.model.add_parameters((4, self.EMB_DIM, 1, 2))
        # self.b3_question = self.model.add_parameters((2, ))
        self.F4_question = self.model.add_parameters((5, self.EMB_DIM, 1, 2))
        self.b4_question = self.model.add_parameters((2, ))

        # dy.dropout(self.F1_question, self.DROPOUT)
        # dy.dropout(self.b1_question, self.DROPOUT)
        # dy.dropout(self.F2_question, self.DROPOUT)
        # dy.dropout(self.b2_question, self.DROPOUT)
        # dy.dropout(self.F3_question, self.DROPOUT)
        # dy.dropout(self.b3_question, self.DROPOUT)
        dy.dropout(self.F4_question, self.DROPOUT)
        dy.dropout(self.b4_question, self.DROPOUT)

        input_size = 1 * (self.EMB_DIM * 2)
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
        dy.dropout(self.W1_query, self.DROPOUT)

        self.W1_question = self.model.add_parameters((self.HIDDEN_DIM, input_size))
        self.bW1_question = self.model.add_parameters((self.HIDDEN_DIM))
        dy.dropout(self.W1_question, self.DROPOUT)


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


    def __convolve2__(self, embeddings, F1, b1, F2, b2, F3, b3, F4, b4, W1, bW1):
        sntlen = len(embeddings)
        emb = dy.concatenate_cols(embeddings)
        x = dy.conv2d_bias(emb, F1, b1, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        x = dy.rectify(x)
        f1 = dy.reshape(x, (self.EMB_DIM * 1 * 2,))

        x = dy.conv2d_bias(emb, F2, b2, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        x = dy.rectify(x)
        f2 = dy.reshape(x, (self.EMB_DIM * 1 * 2,))

        x = dy.conv2d_bias(emb, F3, b3, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        x = dy.rectify(x)
        f3 = dy.reshape(x, (self.EMB_DIM * 1 * 2,))

        x = dy.conv2d_bias(emb, F4, b4, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        x = dy.rectify(x)
        f4 = dy.reshape(x, (self.EMB_DIM * 1 * 2,))

        x = dy.concatenate([f1, f2, f3, f4])
        return W1 * x + bW1


    def __convolve__(self, embeddings, F4, b4, W1, bW1):
        sntlen = len(embeddings)
        emb = dy.concatenate_cols(embeddings)

        # x = dy.conv2d_bias(emb, F3, b3, [1, 1], is_valid=False)
        # x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        # x = dy.rectify(x)
        # f3 = dy.reshape(x, (self.EMB_DIM * 1 * 2,))

        x = dy.conv2d_bias(emb, F4, b4, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [1, sntlen], [1, 1], is_valid=True)
        x = dy.rectify(x)
        f4 = dy.reshape(x, (self.EMB_DIM * 1 * 2,))

        return W1 * f4 + bW1


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
        if self.MODEL[:4] == 'conv':
            query_vec = self.__convolve__(query_embedding,
                                          # self.F1_query,
                                          # self.b1_query,
                                          # self.F2_query,
                                          # self.b2_query,
                                          # self.F3_query,
                                          # self.b3_query,
                                          self.F4_query,
                                          self.b4_query,
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
        if self.MODEL[:4] == 'conv':
            question_vec = self.__convolve__(question_embedding,
                                             # self.F1_question,
                                             # self.b1_question,
                                             # self.F2_question,
                                             # self.b2_question,
                                             # self.F3_question,
                                             # self.b3_question,
                                             self.F4_question,
                                             self.b4_question,
                                             self.W1_question,
                                             self.bW1_question)
        elif self.MODEL == 'lstm':
            question_vec = self.__recur__(question_embedding,
                                           self.fwd_lstm_question,
                                           self.bwd_lstm_question,
                                           self.W1_question,
                                           self.bW1_question)

        return query_vec, question_vec


    def cosine(self, query_vec, question_vec):
        num = dy.transpose(query_vec) * question_vec
        dem1 = dy.sqrt(dy.transpose(query_vec) * query_vec)
        dem2 = dy.sqrt(dy.transpose(question_vec) * question_vec)
        dem = dem1 * dem2

        return dy.cdiv(num, dem)


    def get_loss(self, query, question, label, features={}):
        # forward
        q1, q2 = query['tokens'], question['tokens']
        query_vec, question_vec = self.forward(q1, q2)

        if self.ERROR == 'cosine':
            cosine = self.cosine(query_vec, question_vec)
            if label == 1:
                loss = dy.scalarInput(1) - cosine
            else:
                loss = dy.rectify(cosine-dy.scalarInput(self.m))
        else:
            features = dy.inputTensor(self.__transform__(q1, q2))
            x = dy.concatenate([query_vec, question_vec, features])
            probs = dy.softmax(self.W * x + self.bW)
            loss = -dy.log(dy.pick(probs, label))

        return loss


    def load(self, path):
        self.model.populate(path)


    def fname(self):
        return '_'.join([self.INPUT,
                         str(self.EPOCH),
                         str(self.EMB_DIM),
                         str(self.HIDDEN_DIM),
                         str(self.FEATURE_DIM),
                         self.ERROR,
                         str(self.EARLY_STOP),
                         str(self.BATCH),
                         str(self.DROPOUT),
                         str(self.MODEL),
                         str(self.pretrained)])


    def test(self, testset):
        ranking = {}
        y_real, y_pred = [], []
        for i, qid in enumerate(testset):
            ranking[qid] = []
            percentage = round(float(i+1) / len(testset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')

            query = testset[qid]
            if self.INPUT == 'token':
                q1 = query['tokens']
            else:
                q1 = query['trigrams']
            query_embedding = self.__embed__(q1)

            query_vec = None
            if self.MODEL[:4] == 'conv':
                query_vec = self.__convolve__(query_embedding,
                                              # self.F1_question,
                                              # self.b1_question,
                                              # self.F2_question,
                                              # self.b2_question,
                                              # self.F3_question,
                                              # self.b3_question,
                                              self.F4_question,
                                              self.b4_question,
                                              self.W1_question,
                                              self.bW1_question)
            elif self.MODEL == 'lstm':
                query_vec = self.__recur__(query_embedding,
                                           self.fwd_lstm_query,
                                           self.bwd_lstm_query,
                                           self.W1_query,
                                           self.bW1_query)


            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                rel_question_id = rel_question['id']
                if self.INPUT == 'token':
                    q2 = rel_question['tokens']
                else:
                    q2 = rel_question['trigrams']
                question_embedding = self.__embed__(q2)

                question_vec = None
                if self.MODEL[:4] == 'conv':
                    question_vec = self.__convolve__(question_embedding,
                                                     # self.F1_query,
                                                     # self.b1_query,
                                                     # self.F2_query,
                                                     # self.b2_query,
                                                     # self.F3_query,
                                                     # self.b3_query,
                                                     self.F4_query,
                                                     self.b4_query,
                                                     self.W1_query,
                                                     self.bW1_query)
                elif self.MODEL == 'lstm':
                    question_vec = self.__recur__(question_embedding,
                                                  self.fwd_lstm_question,
                                                  self.bwd_lstm_question,
                                                  self.W1_question,
                                                  self.bW1_question)
                if self.ERROR == 'cosine':
                    score = self.cosine(query_vec, question_vec).value()
                    pred_label = 'true'
                else:
                    # FEATURES
                    features = dy.inputTensor(self.__transform__(q1, q2))

                    x = dy.concatenate([query_vec, question_vec, features])
                    probs = dy.softmax(self.W * x + self.bW)
                    score = dy.pick(probs, 1).value()
                    if score > 0.5:
                        pred_label = 1
                    else:
                        pred_label = 0
                    y_pred.append(pred_label)

                    if rel_question['relevance'] != 'Irrelevant':
                        y_real.append(1)
                    else:
                        y_real.append(0)
                ranking[qid].append((pred_label, score, rel_question_id))
            dy.renew_cg()

        gold = copy.copy(self.devgold)
        map_baseline, map_model = utils.evaluate(gold, ranking)
        f1score = f1_score(y_real, y_pred)
        return map_baseline, map_model, f1score


    def train(self):
        dy.renew_cg()
        trainer = dy.AdadeltaTrainer(self.model)

        # Loggin
        path = self.fname() + '.log'
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
            for i, trainrow in enumerate(self.traindata):
                start = time.time()
                query = { 'id': trainrow['q1_id'], 'tokens': trainrow['q1'] }
                question = { 'id': trainrow['q2_id'], 'tokens': trainrow['q2'] }
                label = trainrow['label']

                features = {}
                if 'features' in trainrow:
                    features = trainrow['features']

                loss = self.get_loss(query, question, label, features)
                losses.append(loss)

                if len(losses) == self.BATCH:
                    loss = dy.esum(losses)
                    _loss = loss.value()
                    closs += _loss
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    # percentage of trainset processed
                    percentage = str(round((float(i+1) / len(self.traindata)) * 100,2)) + '%'
                    # time of epoch processing
                    time_epoch = sum(epoch_timing)
                    if time_epoch > 3600:
                        time_epoch = str(round(time_epoch / 3600, 2)) + ' h'
                    elif time_epoch > 60:
                        time_epoch = str(round(time_epoch / 60, 2)) + ' min'
                    else:
                        time_epoch = str(round(time_epoch, 2)) + ' sec'

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
            map_baseline, map_model, f1score = self.test(self.devset)

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


    # FEATURE EXTRACTION METHODS
    def __transform__(self, q1, q2):
        if type(q1) == list: q1 = ' '.join(q1)
        if type(q2) == list: q2 = ' '.join(q2)

        lcs = features.lcs(re.split('(\W)', q1), re.split('(\W)', q2))
        lcs1 = len(lcs[1].split())
        lcs2 = lcs[0]
        lcsub = features.lcsub(q1, q2)[0]
        jaccard = features.jaccard(q1, q2)
        containment_similarity = features.containment_similarities(q1, q2)
        # greedy_tiling = features.greedy_string_tiling(q1, q2)
        lmprob, trmprob, trlmprob, proctime = self.translation.score(q1.split(), q2.split())

        X = [lcs1, lcsub, jaccard, containment_similarity, lmprob]

        # ngram features
        for n in range(2, 5):
            ngram1 = ' '
            for gram in nltk.ngrams(q1.split(), n):
                ngram1 += 'x'.join(gram) + ' '

            ngram2 = ' '
            for gram in nltk.ngrams(q2.split(), n):
                ngram2 += 'x'.join(gram) + ' '

            lcs = features.lcs(re.split('(\W)', ngram1), re.split('(\W)', ngram2))
            X.append(len(lcs[1].split()))
            # X.append(lcs[0])
            X.append(features.lcsub(ngram1, ngram2)[0])
            X.append(features.jaccard(ngram1, ngram2))
            X.append(features.containment_similarities(ngram1, ngram2))

        return X

if __name__ == '__main__':
    if not os.path.exists(EVALUATION_PATH):
        os.mkdir(EVALUATION_PATH)

    print('Load corpus')
    trainset, devset = load.run()

    ## LSTMs

    properties = {
        'INPUT': 'token',
        'EPOCH': 30,
        'BATCH': 64,
        'EMB_DIM': 300,
        'HIDDEN_DIM': 512,
        'ERROR':'entropy',
        'DROPOUT': 0.2,
        'm': 0.1,
        'EARLY_STOP': 10,
        'MODEL': 'lstm',
        'pretrained_input': True
    }

    siamese = SemevalSiamese(properties, trainset, devset, [])
    siamese.train()

    # CONV
    properties = {
        'INPUT': 'token',
        'EPOCH': 60,
        'BATCH': 32,
        'EMB_DIM': 300,
        'HIDDEN_DIM': 128,
        'ERROR':'entropy',
        'DROPOUT': 0.2,
        'm': 0.1,
        'EARLY_STOP': 20,
        'MODEL': 'conv',
        'pretrained_input': True
    }

    # siamese = SemevalSiamese(properties, trainset, devset, [])
    # siamese.train()