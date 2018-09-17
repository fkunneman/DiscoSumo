__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import copy
import dynet as dy
import ev, metrics
import json
import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import numpy as np
import os
import re
from operator import itemgetter
# ELMo
from allennlp.modules.elmo import Elmo, batch_to_ids

MODEL_PATH='models/models_elmo'
EVALUATION_PATH='results/results_elmo'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

GLOVE_PATH='/home/tcastrof/workspace/glove/glove.6B.300d.txt'
ELMO_PATH='elmo'

def prepare_corpus(indexset):
    for qid in indexset:
        question = indexset[qid]
        q1 = question['subject'] + ' ' + question['body']
        q1 = re.sub(r'[^A-Za-z0-9]+',' ', q1).strip()
        q1 = [w for w in nltk.word_tokenize(q1.lower()) if w not in stop]
        question['tokens'] = q1 + ['eos']

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2 = rel_question['subject']
            if rel_question['body']:
                q2 += ' ' + rel_question['body']
            q2 = re.sub(r'[^A-Za-z0-9]+',' ', q2).strip()
            q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
            rel_question['tokens'] = q2 + ['eos']

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                q2 = rel_comment['text']
                q2 = re.sub(r'[^A-Za-z0-9]+',' ', q2).strip()
                q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
                rel_comment['tokens'] = q2 + ['eos']

    return indexset

def load_glove():
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

def prepare_gold(path):
    ir = ev.read_res_file_aid(path, 'trec')
    return ir

def evaluate(gold, pred):
    for qid in gold:
        gold_sorted = sorted(gold[qid], key = itemgetter(2), reverse = True)
        pred_sorted = pred[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(2), reverse = True)

        gold[qid], pred[qid] = [], []
        for i, row in enumerate(gold_sorted):
            relevant, gold_score, aid = row
            gold[qid].append((relevant, gold_score, aid))

            pred_score = pred_sorted[i][1]
            pred[qid].append((relevant, pred_score, aid))

    for qid in gold:
        # Sort by IR score.
        gold_sorted = sorted(gold[qid], key = itemgetter(1), reverse = True)

        # Sort by SVM prediction score.
        pred_sorted = pred[qid]
        pred_sorted = sorted(pred_sorted, key = itemgetter(1), reverse = True)

        gold[qid] = [rel for rel, score, aid in gold_sorted]
        pred[qid] = [rel for rel, score, aid in pred_sorted]

    map_gold = metrics.map(gold, 10)
    map_pred = metrics.map(pred, 10)
    return map_gold, map_pred

class _Elmo():
    def __init__(self):
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)

    def contextualize(self, sentences):
        token_ids = batch_to_ids(sentences)
        context = self.elmo(token_ids)
        embeddings = context['elmo_representations'][0].detach().tolist()

        sentences = []
        for snt_id, snt in enumerate(context['mask'].tolist()):
            sntcontext = embeddings[snt_id]
            sentence = []
            for w_id, w in enumerate(snt):
                if w_id == 1:
                    sentence.append(sntcontext[w_id])
            sentences.append(sentence)
        return sentences

def prepare_elmo(elmo, indexset, fname):
    contexts = {}
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')

        question = indexset[qid]
        if qid not in contexts:
            contexts[qid] = { 'elmo': [], 'rel_questions': {}, 'rel_comments': {} }
            contexts[qid]['elmo'] = elmo.contextualize(question['tokens'])[0]

        duplicates = question['duplicates']
        for duplicate in duplicates:
            ids, texts = [], []
            rel_question = duplicate['rel_question']
            texts.append(rel_question['tokens'])
            for rel_comment in duplicate['rel_comments']:
                ids.append(rel_comment['id'])
                texts.append(rel_comment['tokens'])

            results = elmo.contextualize(texts)
            contexts[qid]['rel_questions'][rel_question['id']] = results[0]
            for j, result in enumerate(results[1:]):
                contexts[qid]['rel_comments'][ids[j]] = result

        if not os.path.exists(ELMO_PATH):
            os.mkdir(ELMO_PATH)

    json.dump(contexts, open(os.path.join(ELMO_PATH, fname), 'w'), separators=(':'), indent=4)
    return contexts

class Frobenius():
    def __init__(self, trainset, devset, testset):
        print('Preparing trainset...')
        self.trainset = prepare_corpus(trainset)
        print('\nPreparing development set...')
        self.devset = prepare_corpus(devset)
        self.devgold = prepare_gold(GOLD_PATH)

        print('\nInitializing ELMo...')

        if not os.path.exists(ELMO_PATH):
            self.elmo = _Elmo()
            self.trainelmo = prepare_elmo(self.elmo, self.trainset, 'trainvectors.json')
            self.develmo = prepare_elmo(self.elmo, self.devset, 'devvectors.json')
        else:
            self.trainelmo = json.load(open(os.path.join(ELMO_PATH, 'trainvectors.json')))
            self.develmo = json.load(open(os.path.join(ELMO_PATH, 'devvectors.json')))

        self.init()
        self.run(self.devset)


    def init(self):
        dy.renew_cg()
        self.model = dy.Model()

        embeddings, self.voc2id, self.id2voc = load_glove()
        self.lp = self.model.lookup_parameters_from_numpy(embeddings)

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

    def cosine(self, query_vec, question_vec):
        num = dy.transpose(query_vec) * question_vec
        dem1 = dy.sqrt(dy.transpose(query_vec) * query_vec)
        dem2 = dy.sqrt(dy.transpose(question_vec) * question_vec)
        dem = dem1 * dem2

        return dy.cdiv(num, dem)

    def frobenius_norm(self, query, question, typeset='train', use_elmo=True):
        if use_elmo:
            if typeset == 'train':
                elmovectors = self.trainelmo
            elif typeset == 'dev':
                elmovectors = self.develmo
            else:
                elmovectors = {}

            query_id = query['id']
            query_context = list(map(lambda x: dy.inputTensor(np.array(x)), elmovectors[query_id]['elmo']))
            query_emb = [dy.concatenate(list(p)) for p in zip(query['embedding'], query_context)]

            question_id = question['id']
            if question_id in elmovectors[query_id]['rel_questions']:
                question_context = list(map(lambda x: dy.inputTensor(np.array(x)), elmovectors[query_id]['rel_questions'][question_id]))
            else:
                question_context = list(map(lambda x: dy.inputTensor(np.array(x)), elmovectors[query_id]['rel_comments'][question_id]))
            question_emb = [dy.concatenate(list(p)) for p in zip(question['embedding'], question_context)]
        else:
            query_emb = query['embedding']
            question_emb = question['embedding']

        frobenius = 0.0
        for i in range(len(query_emb)):
            for j in range(len(question_emb)):
                cos = dy.rectify(self.cosine(query_emb[i], question_emb[j])).value()
                frobenius += (cos**2)

        return np.sqrt(frobenius)

    def run(self, indexset):
        ranking = {}
        for i, qid in enumerate(indexset):
            ranking[qid] = []
            percentage = round(float(i+1) / len(indexset), 2)
            print('Progress: ', percentage, sep='\t', end='\r')

            query = indexset[qid]
            q1 = query['tokens']
            query_embedding = self.__embed__(q1)

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                rel_question_id = rel_question['id']
                q2 = rel_question['tokens']
                question_embedding = self.__embed__(q2)


                # FEATURES
                query_input = { 'id': qid, 'tokens': q1, 'embedding': query_embedding }
                question_input = { 'id': rel_question_id, 'tokens': q2, 'embedding': question_embedding }
                frobenius =self.frobenius_norm(query=query_input, question=question_input, typeset='dev', use_elmo=False)
                ranking[qid].append(('true', frobenius, rel_question_id))
            dy.renew_cg()

        gold = copy.copy(self.devgold)
        map_baseline, map_model = evaluate(gold, ranking)

        print('MAP Model: ', round(map_model, 2), 'MAP baseline: ', round(map_baseline, 2), sep='\t', end='\n')

if __name__ == '__main__':
    if not os.path.exists(EVALUATION_PATH):
        os.mkdir(EVALUATION_PATH)
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    print('Load corpus')
    trainset, devset = load.run()

    siamese = Frobenius(trainset, devset, [])