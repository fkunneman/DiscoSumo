__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import ev
import dynet as dy
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import re
import time

MODEL_PATH='siamesemodels'
EVALUATION_PATH='siameseresults'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

def get_trigrams(snt):
    trigrams = []
    for word in snt.split():
        word = ['#'] + list(word) + ['#']
        trigrams.extend(list(map(lambda trigram: ''.join(trigram), nltk.ngrams(word, 3))))
    trigrams.append('eos')
    return trigrams

def prepare_corpus(indexset):
    for qid in indexset:
        question = indexset[qid]
        q1 = question['subject'] + ' ' + question['body']
        q1 = re.sub(r'[^A-Za-z0-9]+',' ', q1).strip()
        q1 = [w for w in nltk.word_tokenize(q1.lower()) if w not in stop]
        question['trigrams'] = get_trigrams(' '.join(q1))

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2 = rel_question['subject']
            if rel_question['body']:
                q2 += ' ' + rel_question['body']
            q2 = re.sub(r'[^A-Za-z0-9]+',' ', q2).strip()
            q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
            rel_question['trigrams'] = get_trigrams(' '.join(q2))

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                q2 = rel_comment['text']
                q2 = re.sub(r'[^A-Za-z0-9]+',' ', q2).strip()
                q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
                rel_comment['trigrams'] = get_trigrams(' '.join(q2))

    return indexset

def prepare_traindata(indexset):
    trainset, vocabulary = [], []
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')

        question = indexset[qid]
        q1_trigrams = question['trigrams']
        vocabulary.extend(q1_trigrams)

        trainset.append({
            'q1_trigrams': q1_trigrams,
            'q2_trigrams': q1_trigrams,
            'label':1
        })

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2_trigrams = rel_question['trigrams']
            vocabulary.extend(q2_trigrams)

            if rel_question['relevance'] != 'Irrelevant':
                trainset.append({
                    'q1_trigrams': q1_trigrams,
                    'q2_trigrams': q2_trigrams,
                    'label':1
                })
            else:
                trainset.append({
                    'q1_trigrams': q1_trigrams,
                    'q2_trigrams': q2_trigrams,
                    'label':0
                })

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                if rel_comment['relevance2question'] == 'Bad' and rel_comment['relevance2relquestion'] == 'Bad':
                    q2_trigrams = rel_comment['trigrams']
                    trainset.append({
                        'q1_trigrams': q1_trigrams,
                        'q2_trigrams': q2_trigrams,
                        'label':0
                    })

    vocabulary.append('UNK')
    vocabulary.append('eos')
    vocabulary = list(set(vocabulary))

    id2trigram = {}
    for i, trigram in enumerate(vocabulary):
        id2trigram[i] = trigram

    trigram2id = dict(map(lambda x: (x[1], x[0]), id2trigram.items()))
    return trainset, trigram2id, id2trigram

class SemevalSiamese():
    def __init__(self, properties, trainset, devset, testset):
        print('Preparing trainset...')
        self.trainset = prepare_corpus(trainset)
        self.traindata, self.trigram2id, self.id2trigram = prepare_traindata(self.trainset)
        print('TRAIN DATA SIZE: ', len(self.traindata))
        print('\nPreparing development set...')
        self.devset = prepare_corpus(devset)
        print('\nPreparing test set...')
        self.testset = prepare_corpus(testset)
        print('\nInitializing model...')

        self.MODEL = properties['MODEL']
        self.EPOCH = properties['EPOCH']
        self.BATCH = properties['BATCH']
        self.EMB_DIM = properties['EMB_DIM']
        self.HIDDEN_DIM = properties['HIDDEN_DIM']
        self.ERROR = properties['ERROR']
        self.DROPOUT = properties['DROPOUT']
        self.m = properties['m']
        self.init()


    def init(self):
        VOCAB_SIZE = len(self.trigram2id)

        dy.renew_cg()
        self.model = dy.Model()
        self.lp = self.model.add_lookup_parameters((VOCAB_SIZE, self.EMB_DIM))

        if self.MODEL[:4] == 'conv':
            self.init_conv()
        elif self.MODEL == 'lstm':
            self.init_lstm()

        if self.ERROR == 'entropy':
            self.W = self.model.add_parameters((2, self.HIDDEN_DIM*2))
            self.bW = self.model.add_parameters((2))


    def init_conv(self):
        # QUERY
        # 4 filters bi- tri- four- and five-grams with 2 feature maps
        self.F1_query = self.model.add_parameters((2, self.EMB_DIM, 1, 2))
        self.b1_query = self.model.add_parameters((2, ))
        self.F2_query = self.model.add_parameters((3, self.EMB_DIM, 1, 2))
        self.b2_query = self.model.add_parameters((2, ))
        self.F3_query = self.model.add_parameters((4, self.EMB_DIM, 1, 2))
        self.b3_query = self.model.add_parameters((2, ))
        self.F4_query = self.model.add_parameters((5, self.EMB_DIM, 1, 2))
        self.b4_query = self.model.add_parameters((2, ))

        dy.dropout(self.F1_query, self.DROPOUT)
        dy.dropout(self.b1_query, self.DROPOUT)
        dy.dropout(self.F2_query, self.DROPOUT)
        dy.dropout(self.b2_query, self.DROPOUT)
        dy.dropout(self.F3_query, self.DROPOUT)
        dy.dropout(self.b3_query, self.DROPOUT)
        dy.dropout(self.F4_query, self.DROPOUT)
        dy.dropout(self.b4_query, self.DROPOUT)

        # CANDIDATE QUESTION
        # 4 filters bi- tri- four- and five-grams with 2 feature maps
        self.F1_question = self.model.add_parameters((2, self.EMB_DIM, 1, 2))
        self.b1_question = self.model.add_parameters((2, ))
        self.F2_question = self.model.add_parameters((3, self.EMB_DIM, 1, 2))
        self.b2_question = self.model.add_parameters((2, ))
        self.F3_question = self.model.add_parameters((4, self.EMB_DIM, 1, 2))
        self.b3_question = self.model.add_parameters((2, ))
        self.F4_question = self.model.add_parameters((5, self.EMB_DIM, 1, 2))
        self.b4_question = self.model.add_parameters((2, ))

        dy.dropout(self.F1_question, self.DROPOUT)
        dy.dropout(self.b1_question, self.DROPOUT)
        dy.dropout(self.F2_question, self.DROPOUT)
        dy.dropout(self.b2_question, self.DROPOUT)
        dy.dropout(self.F3_question, self.DROPOUT)
        dy.dropout(self.b3_question, self.DROPOUT)
        dy.dropout(self.F4_question, self.DROPOUT)
        dy.dropout(self.b4_question, self.DROPOUT)

        input_size = 4 * (self.EMB_DIM * 2)
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
        trigram_question = []
        trigram_index = []
        for trigram in text:
            if trigram not in trigram_question:
                trigram_question.append(trigram)
                try:
                    _id = self.trigram2id[trigram]
                except:
                    _id = self.trigram2id['UNK']
                trigram_index.append(_id)

        return list(map(lambda idx: self.lp[idx], trigram_index))


    def __convolve__(self, embeddings, F1, b1, F2, b2, F3, b3, F4, b4, W1, bW1):
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

        lpidx = self.lp[self.trigram2id['eos']]
        fwd_vectors = self.run_lstm(fwd_lstm.initial_state().add_input(lpidx), embeddings)
        bwd_vectors = self.run_lstm(bwd_lstm.initial_state().add_input(lpidx), embeddings_rev)
        bwd_vectors = list(reversed(bwd_vectors))

        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        x = dy.average(vectors)
        return W1 * x + bW1


    def forward(self, query, question):
        query_embedding = self.__embed__(query)
        query_vec = None
        if self.MODEL[:4] == 'conv':
            query_vec = self.__convolve__(query_embedding,
                                          self.F1_query,
                                          self.b1_query,
                                          self.F2_query,
                                          self.b2_query,
                                          self.F3_query,
                                          self.b3_query,
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
                                             self.F1_question,
                                             self.b1_question,
                                             self.F2_question,
                                             self.b2_question,
                                             self.F3_question,
                                             self.b3_question,
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


    def get_classification_loss(self, query_vec, question_vec, label):
        x = dy.concatenate([query_vec, question_vec])
        probs = dy.softmax(self.W * x + self.bW)

        return -dy.log(dy.pick(probs, label))


    def cosine(self, query_vec, question_vec):
        num = dy.transpose(query_vec) * question_vec
        dem1 = dy.sqrt(dy.transpose(query_vec) * query_vec)
        dem2 = dy.sqrt(dy.transpose(question_vec) * question_vec)
        dem = dem1 * dem2

        return dy.cdiv(num, dem)


    def get_loss(self, query, question, label):
        query_vec, question_vec = self.forward(query, question)

        cosine = self.cosine(query_vec, question_vec)
        if label == 1:
            loss = dy.scalarInput(1) - cosine
        else:
            loss = dy.rectify(cosine-dy.scalarInput(self.m))
        return loss


    def load(self, path):
        self.model.populate(path)


    def test(self, testset):
        def rank(ranking):
            _ranking = []
            for i, q in enumerate(sorted(ranking, key=lambda x: x[1], reverse=True)):
                _ranking.append({'Answer_ID':q[0], 'SCORE':q[1], 'RANK':i+1})
            return _ranking

        ranking = {}
        for i, qid in enumerate(testset):
            ranking[qid] = []
            percentage = round(float(i+1) / len(testset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')
            question = testset[qid]
            trigrams = question['trigrams']

            embedding = self.__embed__(trigrams)
            query_vec = None
            if self.MODEL[:4] == 'conv':
                query_vec = self.__convolve__(embedding,
                                              self.F1_question,
                                              self.b1_question,
                                              self.F2_question,
                                              self.b2_question,
                                              self.F3_question,
                                              self.b3_question,
                                              self.F4_question,
                                              self.b4_question,
                                              self.W1_question,
                                              self.bW1_question)
            elif self.MODEL == 'lstm':
                query_vec = self.__recur__(embedding,
                                           self.fwd_lstm_query,
                                           self.bwd_lstm_query,
                                           self.W1_query,
                                           self.bW1_query)


            duplicates = question['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                rel_question_id = rel_question['id']
                trigrams = rel_question['trigrams']

                embedding = self.__embed__(trigrams)
                question_vec = None
                if self.MODEL[:4] == 'conv':
                    question_vec = self.__convolve__(embedding,
                                                     self.F1_query,
                                                     self.b1_query,
                                                     self.F2_query,
                                                     self.b2_query,
                                                     self.F3_query,
                                                     self.b3_query,
                                                     self.F4_query,
                                                     self.b4_query,
                                                     self.W1_query,
                                                     self.bW1_query)
                elif self.MODEL == 'lstm':
                    question_vec = self.__recur__(embedding,
                                                  self.fwd_lstm_question,
                                                  self.bwd_lstm_question,
                                                  self.W1_question,
                                                  self.bW1_question)
                if self.ERROR == 'cosine':
                    score = self.cosine(query_vec, question_vec).value()
                else:
                    x = dy.concatenate([query_vec, question_vec])
                    probs = dy.softmax(self.W * x + self.bW)
                    score = dy.pick(probs, 1).value()
                ranking[qid].append((rel_question_id, score))
            ranking[qid] = rank(ranking[qid])
            dy.renew_cg()

        fname = '_'.join([str(self.EPOCH),str(self.EMB_DIM),str(self.HIDDEN_DIM),str(self.BATCH),str(self.DROPOUT), str(self.MODEL)]) + '.pred'
        PRED_PATH = os.path.join(EVALUATION_PATH, fname)
        load.save(ranking, PRED_PATH)

        map_model, map_baseline = ev.eval_rerankerV2(GOLD_PATH, PRED_PATH)
        return map_model, map_baseline


    def train(self):
        trainer = dy.AdadeltaTrainer(self.model)

        # Loggin
        path = '_'.join([str(self.EPOCH),
                         str(self.EMB_DIM),
                         str(self.HIDDEN_DIM),
                         self.ERROR,
                         str(self.BATCH),
                         str(self.DROPOUT),
                         self.MODEL]) + '.log'
        f = open(os.path.join(MODEL_PATH, path), 'w')

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
                query = trainrow['q1_trigrams']
                question = trainrow['q2_trigrams']
                label = trainrow['label']

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
            # print('Train evaluation...')
            # self.test(list(self.trainset.keys())[:500])
            print('Dev evaluation...')
            f.write('Dev evaluation...\n')
            map_model, map_baseline = self.test(self.devset)
            print('MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), sep='\t', end='\n')
            f.write('\t'.join(['MAP Model: ', str(round(map_model, 2)), 'MAP baseline: ', str(round(map_baseline, 2)), '\n']))

            epoch_timing = []
            if map_model > best:
                best = map_model
                early = 0
                path = '_'.join([str(self.EPOCH),
                                 str(self.EMB_DIM),
                                 str(self.HIDDEN_DIM),
                                 self.ERROR,
                                 str(self.BATCH),
                                 str(self.DROPOUT),
                                 self.MODEL]) + '.dy'
                self.model.save(os.path.join(MODEL_PATH, path))
            else:
                early += 1

            if early == 5:
                break
        f.close()

if __name__ == '__main__':
    if not os.path.exists(EVALUATION_PATH):
        os.mkdir(EVALUATION_PATH)
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    print('Load corpus')
    trainset, devset = load.run()

    properties = {
        'EPOCH': 40,
        'BATCH': 64,
        'EMB_DIM': 256,
        'HIDDEN_DIM': 512,
        'DROPOUT': 0.3,
        'ERROR':'cosine',
        'm': 0.1,
        'MODEL': 'lstm'
    }
    path = '_'.join([str(properties['EPOCH']),
                     str(properties['EMB_DIM']),
                     str(properties['HIDDEN_DIM']),
                     properties['ERROR'],
                     str(properties['BATCH']),
                     str(properties['DROPOUT']),
                     properties['MODEL']]) + '.dy'

    # siamese = SemevalSiamese(properties, trainset, devset, [])
    # siamese.train()

    properties = {
        'EPOCH': 40,
        'BATCH': 128,
        'EMB_DIM': 128,
        'HIDDEN_DIM': 128,
        'ERROR':'cosine',
        'DROPOUT': 0.3,
        'm': 0.1,
        'MODEL': 'conv'
    }
    path = '_'.join([str(properties['EPOCH']),
                     str(properties['EMB_DIM']),
                     str(properties['HIDDEN_DIM']),
                     properties['ERROR'],
                     str(properties['BATCH']),
                     str(properties['DROPOUT']),
                     properties['MODEL']]) + '.dy'

    siamese = SemevalSiamese(properties, trainset, devset, [])
    siamese.train()