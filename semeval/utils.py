__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import ev, metrics
import json
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import re
from operator import itemgetter

from gensim import corpora

def get_trigrams(snt):
    trigrams = []
    for word in snt.split():
        word = ['#'] + list(word) + ['#']
        trigrams.extend(list(map(lambda trigram: ''.join(trigram), nltk.ngrams(word, 3))))
    trigrams.append('eos')
    return trigrams

def parse(question, corenlp, props):
    try:
        out = json.loads(corenlp.annotate(question, properties=props))

        tokens = []
        sentences = '(SENTENCES '
        for snt in out['sentences']:
            tokens.extend(map(lambda x: x['originalText'], snt['tokens']))
            sentences += snt['parse'].replace('\n', '') + ' '
        sentences = sentences.strip()
        sentences += ')'
    except:
        print('parsing error...')
        tokens, sentences = '', '()'
    return ' '.join(tokens), sentences

def prepare_corpus(indexset, corenlp, props):
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')
        question = indexset[qid]
        q1 = question['subject'] + ' ' + question['body']
        tokens, question['tree'] = parse(q1, corenlp, props)
        q1 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
        q1 = [w for w in q1.lower().split() if w not in stop]
        question['tokens'] = q1 + ['eos']
        question['trigrams'] = get_trigrams(' '.join(q1))

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2 = rel_question['subject']
            if rel_question['body']:
                q2 += ' ' + rel_question['body']
            tokens, rel_question['tree'] = parse(q2, corenlp, props)
            q2 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
            q2 = [w for w in q2.lower().split() if w not in stop]
            rel_question['tokens'] = q2 + ['eos']
            rel_question['trigrams'] = get_trigrams(' '.join(q2))

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                q2 = rel_comment['text']
                tokens, rel_comment['tree'] = parse(q2, corenlp, props)
                q2 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
                q2 = [w for w in q2.lower().split() if w not in stop]
                rel_comment['tokens'] = q2 + ['eos']
                rel_comment['trigrams'] = get_trigrams(' '.join(q2))

    return indexset

def prepare_traindata(indexset, unittype='token'):
    trainset, vocabulary = [], []

    vocquestions = []
    rel_questions = []
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')

        question = indexset[qid]
        q1_tree = question['tree']
        if unittype == 'token':
            q1 = question['tokens']
        else:
            q1 = question['trigrams']
        vocquestions.append(q1)
        vocabulary.extend(q1)

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2_tree = rel_question['tree']
            if unittype == 'token':
                q2 = rel_question['tokens']
            else:
                q2 = rel_question['trigrams']
            vocquestions.append(q2)
            vocabulary.extend(q2)

            if rel_question['relevance'] != 'Irrelevant':
                trainset.append({
                    'q1_id': qid,
                    'q1': q1,
                    'q1_tree': q1_tree,
                    'q2_id': rel_question['id'],
                    'q2': q2,
                    'q2_tree': q2_tree,
                    'label':1
                })
            else:
                trainset.append({
                    'q1_id': qid,
                    'q1': q1,
                    'q1_tree': q1_tree,
                    'q2_id': rel_question['id'],
                    'q2': q2,
                    'q2_tree': q2_tree,
                    'label':0
                })

            # Related questions to augment the corpus
            # rel_questions.append((qid, q2))
            #
            # rel_comments = duplicate['rel_comments']
            # for rel_comment in rel_comments:
            #     if rel_comment['relevance2question'] == 'Bad' and rel_comment['relevance2relquestion'] == 'Bad':
            #         if unittype == 'token':
            #             q2 = rel_comment['tokens']
            #         else:
            #             q2 = rel_comment['trigrams']
            #         trainset.append({
            #             'q1_id': qid,
            #             'q1': q1,
            #             'q2_id': rel_comment['id'],
            #             'q2': q2,
            #             'label':0
            #         })

    vocabulary.append('UNK')
    vocabulary.append('eos')
    vocabulary = list(set(vocabulary))

    id2voc = {}
    for i, trigram in enumerate(vocabulary):
        id2voc[i] = trigram

    voc2id = dict(map(lambda x: (x[1], x[0]), id2voc.items()))

    vocabulary = corpora.Dictionary(vocquestions)
    return trainset, voc2id, id2voc, vocabulary

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