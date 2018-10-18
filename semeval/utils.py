__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/semeval/evaluation/MAP_scripts')
import copy
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
    tokens, lemmas, pos = [], [], []
    try:
        out = corenlp.annotate(question, properties=props)
        out = json.loads(out)

        trees = '(SENTENCES '
        for snt in out['sentences']:
            tokens.extend(map(lambda x: x['originalText'], snt['tokens']))
            lemmas.extend(map(lambda x: x['lemma'], snt['tokens']))
            pos.extend(map(lambda x: x['pos'], snt['tokens']))
            trees += snt['parse'].replace('\n', '') + ' '
        trees = trees.strip()
        trees += ')'
    except:
        print('parsing error...')
        tokens, trees = '', '()'
    return ' '.join(tokens), trees, ' '.join(lemmas), pos

def parse_tree(tree, token2lemma={}):
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

def prepare_corpus(indexset, corenlp, props):
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')
        question = indexset[qid]
        q1 = copy.copy(question['subject'])
        tokens, question['subj_tree'], lemmas, pos = parse(q1, corenlp, props)
        # question['subj_tokens_full'] =  [w for w in tokens.lower().split()]
        # question['subj_lemmas_full'] =  [w for w in lemmas.lower().split()]
        # question['subj_pos_full'] =  pos
        # q1 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
        # q1 = [w for w in q1.lower().split() if w not in stop]
        # question['subj_tokens'] = q1 + ['eos']
        # question['subj_trigrams'] = get_trigrams(' '.join(q1))

        q1 = question['subject'] + ' ' + question['body']
        tokens, question['tree'], lemmas, pos = parse(q1, corenlp, props)
        question['tokens'] = [w for w in tokens.lower().split()]
        question['lemmas'] = [w for w in lemmas.lower().split()]
        question['pos'] = pos
        q1 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
        q1 = [w for w in q1.lower().split() if w not in stop]
        question['tokens_proc'] = q1
        question['trigrams'] = get_trigrams(' '.join(q1))

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            q2 = copy.copy(rel_question['subject'])
            tokens, rel_question['subj_tree'], lemmas, pos = parse(q2, corenlp, props)
            # rel_question['subj_tokens_full'] = [w for w in tokens.lower().split()]
            # rel_question['subj_lemmas_full'] = [w for w in lemmas.lower().split()]
            # rel_question['subj_pos_full'] = pos
            # q2 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
            # q2 = [w for w in q2.lower().split() if w not in stop]
            # rel_question['subj_tokens'] = q2 + ['eos']
            # rel_question['subj_trigrams'] = get_trigrams(' '.join(q2))

            q2 = copy.copy(rel_question['subject'])
            if rel_question['body']:
                q2 += ' ' + rel_question['body']
            tokens, rel_question['tree'], lemmas, pos = parse(q2, corenlp, props)
            rel_question['tokens'] = [w for w in tokens.lower().split()]
            rel_question['lemmas'] = [w for w in lemmas.lower().split()]
            rel_question['pos'] = pos
            q2 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
            q2 = [w for w in q2.lower().split() if w not in stop]
            rel_question['tokens_proc'] = q2
            rel_question['trigrams'] = get_trigrams(' '.join(q2))

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                q2 = rel_comment['text']
                tokens, rel_comment['tree'], lemmas, pos = parse(q2, corenlp, props)
                rel_comment['tokens'] = [w for w in tokens.lower().split()]
                rel_comment['lemmas'] = [w for w in lemmas.lower().split()]
                rel_comment['pos'] = pos
                q2 = re.sub(r'[^A-Za-z0-9]+',' ', tokens).strip()
                q2 = [w for w in q2.lower().split() if w not in stop]
                rel_comment['tokens_proc'] = q2
                rel_comment['trigrams'] = get_trigrams(' '.join(q2))

    return indexset

def prepare_traindata(indexset):
    trainset, vocabulary = [], []

    vocquestions = []
    for i, qid in enumerate(indexset):
        percentage = str(round((float(i+1) / len(indexset)) * 100, 2)) + '%'
        print('Process: ', percentage, end='\r')

        question = indexset[qid]
        subj_q1_tree = question['subj_tree']
        q1_tree = question['tree']
        q1_pos = question['pos']
        q1_lemmas = question['lemmas']
        q1_full = question['tokens']
        q1 = question['tokens_proc']

        vocquestions.append(q1)
        vocabulary.extend(q1)

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            subj_q2_tree = rel_question['subj_tree']
            q2_tree = rel_question['tree']
            q2_pos = rel_question['pos']
            q2_lemmas = rel_question['lemmas']
            q2_full = rel_question['tokens']
            q2 = rel_question['tokens_proc']
            vocquestions.append(q2)
            vocabulary.extend(q2)

            # Related questions to augment the corpus
            comments = []

            rel_comments = duplicate['rel_comments']
            for rel_comment in rel_comments:
                q2 = rel_comment['tokens_proc']

                comments.append({
                    'id': rel_comment['id'],
                    'tokens': q2,
                })

            label = 0
            if rel_question['relevance'] != 'Irrelevant':
                label = 1
            trainset.append({
                'q1_id': qid,
                'q1': q1,
                'q1_full': q1_full,
                'q1_tree': q1_tree,
                'subj_q1_tree': subj_q1_tree,
                'q1_lemmas': q1_lemmas,
                'q1_pos': q1_pos,
                'q2_id': rel_question['id'],
                'q2': q2,
                'q2_full': q2_full,
                'q2_tree': q2_tree,
                'subj_q2_tree': subj_q2_tree,
                'q2_lemmas': q2_lemmas,
                'q2_pos': q2_pos,
                'comments': comments,
                'label':label
            })

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