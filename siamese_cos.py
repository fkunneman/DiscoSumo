__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/cqadupstack/CQADupStack')
import query_cqadupstack as qcqa
import os
import json
import nltk
import dynet as dy
import re
import time

from random import shuffle

CORPUS_PATH = '/home/tcastrof/Question/cqadupstack'
CATEGORY = 'android'

QUESTION_TYPE='title_body'

def load_corpus():
    o = qcqa.load_subforum(os.path.join(CORPUS_PATH, CATEGORY+'.zip'))
    testset, develset, indexset = o.split_for_retrieval()
    return o, indexset, develset, testset

def remove_long_tokens(snt):
    _snt = []
    for word in snt.split():
        if len(word) <= 20:
            _snt.append(word)
    return ' '.join(_snt)

def get_trigrams(snt):
    trigrams = []
    for word in snt.split():
        word = ['#'] + list(word) + ['#']
        trigrams.extend(list(map(lambda trigram: ''.join(trigram), nltk.ngrams(word, 3))))
    return trigrams

def prepare_corpus(o, indexset):
    procset = []
    for i, idx in enumerate(indexset):
        try:
            print('Question number: ', i, 'Question ID: ', idx, end='\r')
            # retrieve question
            title_body = o.get_post_title_and_body(idx)
            # removing stopwords and stemming it
            q = o.perform_cleaning(title_body, remove_stopwords=True, remove_punct=False, stem=True)
            # removing punctuation (better than in nltk)
            q = re.sub(r'[^\w\s][ ]*','', q).strip()
            # removing tokens greater than 20
            q = remove_long_tokens(q)

            if len(q.split()) > 0:
                q_trigrams = get_trigrams(q)
                procset.append({
                    'id': idx,
                    'q': q,
                    'q_trigrams': q_trigrams
                })
        except:
            print('\nQuestion error')
    return procset

def prepare_trainset(o, procset):
    trainset = []
    for i, row in enumerate(procset):
        idx = row['id']
        print('Question number: ', i, 'Question ID: ', idx, end='\r')
        q1 = row['q']
        q1_trigrams = row['q_trigrams']

        similar = 0
        for j in range(i+1, len(procset)):
            pair = procset[j]['id']
            if o.get_true_label(idx, pair) != 'nodup':
                q2 = procset[j]['q']
                q2_trigrams = procset[j]['q_trigrams']
                trainset.append({
                    'q1': q1,
                    'q1_trigrams': q1_trigrams,
                    'q2': q2,
                    'q2_trigrams': q2_trigrams,
                    'label':1
                })
                similar += 1
        # same number of no duplicates for each element of the training set
        aux = 0
        shuffle(procset)
        for j in range(i+1, len(procset)):
            pair = procset[j]['id']
            if o.get_true_label(idx, pair) == 'nodup':
                q2 = procset[j]['q']
                q2_trigrams = procset[j]['q_trigrams']
                trainset.append({
                    'q1': q1,
                    'q1_trigrams': q1_trigrams,
                    'q2': q2,
                    'q2_trigrams': q2_trigrams,
                    'label':0
                })
                aux += 1
            if similar == 0 and aux == 5: break
            elif aux == similar: break
    return trainset

def prepare_devset(o, trainprocset, devprocset):
    devset = []
    for i, row in enumerate(devprocset):
        idx = row['id']
        print('Question number: ', i, 'Question ID: ', idx, end='\r')
        q1 = row['q']
        q1_trigrams = row['q_trigrams']

        similar = 0
        for j, trainrow in enumerate(trainprocset):
            pair = trainprocset[j]['id']
            if o.get_true_label(idx, pair) != 'nodup':
                q2 = trainrow['q']
                q2_trigrams = trainprocset[j]['q_trigrams']
                devset.append({
                    'q1': q1,
                    'q1_trigrams': q1_trigrams,
                    'q2': q2,
                    'q2_trigrams': q2_trigrams,
                    'label':1
                })
                similar += 1
        # same number of no duplicates for each element of the training set
        aux = 0
        shuffle(trainprocset)
        for j, trainrow in enumerate(trainprocset):
            pair = trainprocset[j]['id']
            if o.get_true_label(idx, pair) == 'nodup':
                q2 = trainrow['q']
                q2_trigrams = trainprocset[j]['q_trigrams']
                devset.append({
                    'q1': q1,
                    'q1_trigrams': q1_trigrams,
                    'q2': q2,
                    'q2_trigrams': q2_trigrams,
                    'label':0
                })
                aux += 1
            if similar == 0 and aux == 3: break
            elif aux == similar: break
    return devset

def prepare_vocabulary(trainset):
    vocabulary = []
    for i, row in enumerate(trainset):
        percentage = str(round((float(i+1) / len(trainset)) * 100,2)) + '%'
        print('Process: ', percentage, end='\r')
        trigrams = row['q1_trigrams']
        vocabulary.extend(trigrams)
        trigrams = row['q2_trigrams']
        vocabulary.extend(trigrams)

    vocabulary.append('UNK')
    vocabulary = list(set(vocabulary))

    id2trigram = {}
    for i, trigram in enumerate(vocabulary):
        id2trigram[i] = trigram

    trigram2id = dict(map(lambda x: (x[1], x[0]), id2trigram.items()))
    return trigram2id, id2trigram

class SiameseNNs():
    def __init__(self, properties):
        print('Loading corpus...')
        self.o, self.indexset, self.develset, self.testset = load_corpus()
        print('Preparing trainset...')
        self.procset = prepare_corpus(self.o, self.indexset)
        if os.path.exists('trainset.json'):
            self.trainset = json.load(open('trainset.json'))
        else:
            self.trainset = prepare_trainset(self.o, self.procset)
            json.dump(self.trainset, open('trainset.json', 'w'))
        print('\nPreparing vocabulary...')
        self.trigram2id, self.id2trigram = prepare_vocabulary(self.trainset)
        print('\nPreparing development set...')
        self.devset = prepare_corpus(self.o, self.develset)
        print('\nPreparing test set...')
        self.testset = prepare_corpus(self.o, self.testset)
        print('\nInitializing model...')

        self.EPOCH = properties['EPOCH']
        self.BATCH = properties['BATCH']
        self.EMB_DIM = properties['EMB_DIM']
        self.HIDDEN_DIM = properties['HIDDEN_DIM']
        self.DROPOUT = properties['DROPOUT']
        self.m = properties['m']
        self.init()

    def init(self):
        VOCAB_SIZE = len(self.trigram2id)

        dy.renew_cg()
        self.model = dy.Model()
        self.lp = self.model.add_lookup_parameters((VOCAB_SIZE, self.EMB_DIM))

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


    def __encode__(self, embeddings, F1, b1, F2, b2, F3, b3, F4, b4, W1, bW1):
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


    def forward(self, query, question):
        query_embedding = self.__embed__(query)
        query_vec = self.__encode__(query_embedding,
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

        question_embedding = self.__embed__(question)
        question_vec = self.__encode__(question_embedding,
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
        return query_vec, question_vec


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
        train2vec = {}
        for i, trainrow in enumerate(self.procset):
            idx = trainrow['id']
            trigrams = trainrow['q_trigrams']
            embedding = self.__embed__(trigrams)
            train2vec[idx] = self.__encode__(embedding,
                                             self.F1_question,
                                             self.b1_question,
                                             self.F2_question,
                                             self.b2_question,
                                             self.F3_question,
                                             self.b3_question,
                                             self.F4_question,
                                             self.b4_question,
                                             self.W1_question,
                                             self.bW1_question).npvalue()
            if i % self.BATCH == 0:
                dy.renew_cg()

        ranking = {}
        for i, testrow in enumerate(testset):
            percentage = round(float(i+1) / len(testset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')
            query_idx = testrow['id']
            ranking[query_idx] = []

            trigrams = testrow['q_trigrams']
            embedding = self.__embed__(trigrams)
            query = self.__encode__(embedding,
                                    self.F1_query,
                                    self.b1_query,
                                    self.F2_query,
                                    self.b2_query,
                                    self.F3_query,
                                    self.b3_query,
                                    self.F4_query,
                                    self.b4_query,
                                    self.W1_query,
                                    self.bW1_query).npvalue()

            for j, question_idx in enumerate(train2vec.keys()):
                query_vec = dy.inputTensor(query)
                question_vec = dy.inputTensor(train2vec[question_idx])
                cos = self.cosine(query_vec, question_vec).value()
                ranking[query_idx].append((question_idx, cos))

                if j % self.BATCH == 0:
                    dy.renew_cg()


        with open('ranking_with_cos.txt', 'w') as f:
            for query_idx in ranking:
                f.write(query_idx)
                f.write(' ')
                r = list(map(lambda x: str(x[0]) + '_' + str(x[1]), sorted(ranking[query_idx], key=lambda x: x[1], reverse=True)))[:10]
                f.write(' '.join(r))
                f.write(' <br />\n')

        with open('ranking.txt', 'w') as f:
            for query_idx in ranking:
                f.write(query_idx)
                f.write(' ')
                r = list(map(lambda x: x[0], sorted(ranking[query_idx], key=lambda x: x[1], reverse=True)))[:10]
                f.write(' '.join(r))
                f.write(' <br />\n')

        path = 'ranking.txt'
        _map = self.o.mean_average_precision(path)
        precision = self.o.average_precision_at(path)
        print('Mean Averaged Precision (MAP): ', round(_map, 2), 'Precision: ', round(precision, 2), sep='\t', end='\n')
        return _map, precision


    def train(self):
        trainer = dy.AdadeltaTrainer(self.model)

        epoch_timing = []
        early = 0.0
        best = -1
        for epoch in range(self.EPOCH):
            print('\n')
            dy.renew_cg()
            losses = []
            closs = 0
            batch_timing = []
            for i, trainrow in enumerate(self.trainset):
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
                    percentage = str(round((float(i+1) / len(self.trainset)) * 100,2)) + '%'
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

            print("\nEpoch: {0} \t\t Loss: {1} \t\t Best: {2}".format(epoch+1, round(closs/self.BATCH, 2), round(best, 2)))

            print('Train evaluation...')
            self.test(self.procset[:500])
            print('Dev evaluation...')
            _map, precision = self.test(self.devset[:500])

            epoch_timing = []
            if _map > best:
                best = _map
                early = 0
                path = '_'.join([str(self.EPOCH),str(self.EMB_DIM),str(self.HIDDEN_DIM),str(self.BATCH),str(self.DROPOUT)]) + '.dy'
                self.model.save(path)
            else:
                early += 1

            if early == 20:
                break


if __name__ == '__main__':
    properties = {
        'EPOCH': 40,
        'BATCH': 128,
        'EMB_DIM': 128,
        'HIDDEN_DIM': 128,
        'DROPOUT': 0.4,
        'm': 0.1
    }
    path = '_'.join([str(properties['EPOCH']), str(properties['EMB_DIM']), str(properties['HIDDEN_DIM']), str(properties['BATCH']), str(properties['DROPOUT'])]) + '.dy'

    siamese = SiameseNNs(properties)
    siamese.train()

    print('Testing...')
    siamese.load(path)
    siamese.test(siamese.testset)
