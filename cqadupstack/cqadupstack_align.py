__author__='thiagocastroferreira'

import sys
sys.path.append('/home/tcastrof/Question/cqadupstack/CQADupStack')
import query_cqadupstack as qcqa

import os
import re

CORPUS_PATH = '/home/tcastrof/Question/cqadupstack'
CATEGORY = 'android'

def load_corpus():
    o = qcqa.load_subforum(os.path.join(CORPUS_PATH, CATEGORY + '.zip'))
    testset, develset, indexset = o.split_for_retrieval()
    return o, indexset, develset, testset

def remove_long_tokens(snt):
    _snt = []
    for word in snt.split():
        if len(word) <= 20:
            _snt.append(word)
    return ' '.join(_snt)

def prepare_questions(o, indexset):
    trainset = []
    for i, idx in enumerate(indexset):
        try:
            print('Question number: ', i, 'Question ID: ', idx, end='\r')
            # retrieve question
            title_body = o.get_post_title_and_body(idx)
            # removing stopwords and stemming it
            q1 = o.perform_cleaning(title_body, remove_stopwords=True, remove_punct=False, stem=True)
            # removing punctuation (better than in nltk)
            q1 = re.sub(r'[^\w\s][ ]*','', q1).strip()
            # removing tokens greater than 20
            q1 = remove_long_tokens(q1)

            if len(q1.split()) > 2:
                pairs = o.get_duplicates(idx)
                pairs.extend(o.get_related(idx))

                for pair in pairs:
                    title_body = o.get_post_title_and_body(pair)
                    # removing stopwords and stemming it
                    q2 = o.perform_cleaning(title_body, remove_stopwords=True, remove_punct=False, stem=True)
                    # removing punctuation (better than in nltk)
                    q2 = re.sub(r'[^\w\s][ ]*','', q2).strip()
                    # removing tokens greater than 20
                    q2 = remove_long_tokens(q2)

                    if len(q2.split()) > 2:
                        trainset.append({
                            'source': q1,
                            'target': q2
                        })
                        trainset.append({
                            'source': q2,
                            'target': q1
                        })
        except:
            print('Question Error')

    return trainset

def save(trainset):
    path = os.path.join(CORPUS_PATH, CATEGORY, 'translation')
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path, CATEGORY+'.de'), 'w') as f:
        f.write('\n'.join(map(lambda x: x['source'], trainset)))

    with open(os.path.join(path, CATEGORY+'.en'), 'w') as f:
        f.write('\n'.join(map(lambda x: x['target'], trainset)))

if __name__ == '__main__':
    print('Load corpus')
    o, indexset, develset, testset = load_corpus()
    print('Preparing training questions for alignment')
    trainset = prepare_questions(o, indexset)
    print('Saving Parallel data')
    save(trainset)