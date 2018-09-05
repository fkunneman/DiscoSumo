__author__='thiagocastroferreira'

import load
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import re

TRANSLATION_PATH='translation'

def prepare_questions(procset):
    trainset = []
    for i, qid in enumerate(procset):
        question = procset[qid]
        percentage = round(float(i+1) / len(procset), 2)
        print('Progress: ', percentage, sep='\t', end='\r')

        q1 = question['subject'] + ' ' + question['body']
        # tokenizing and removing punctuation / stopwords
        q1 = re.sub(r'[^\w\s]',' ', q1).strip()
        q1 = [w for w in nltk.word_tokenize(q1.lower()) if w not in stop]
        q1 = ' '.join(q1)

        duplicates = question['duplicates']
        for duplicate in duplicates:
            rel_question = duplicate['rel_question']
            # if related question is not irrelevant
            if rel_question['relevance'] != 'Irrelevant':
                q2 = rel_question['subject']
                if rel_question['body']:
                    q2 += ' ' + rel_question['body']
                # tokenizing and removing punctuation / stopwords
                q2 = re.sub(r'[^\w\s]',' ', q2).strip()
                q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
                q2 = ' '.join(q2)

                trainset.append({
                    'source': q1,
                    'target': q2
                })
                trainset.append({
                    'source': q2,
                    'target': q1
                })

                rel_comments = duplicate['rel_comments']
                for rel_comment in rel_comments:
                    # if comment is not bad
                    if rel_comment['relevance2question'] != 'Bad' and rel_comment['relevance2relquestion'] != 'Bad':
                        q2 = re.sub(r'[^\w\s]',' ', rel_comment['text'].lower()).strip()
                        q2 = [w for w in nltk.word_tokenize(q2.lower()) if w not in stop]
                        q2 = ' '.join(q2)

                        trainset.append({
                            'source': q1,
                            'target': q2
                        })
                        trainset.append({
                            'source': q2,
                            'target': q1
                        })
    return trainset

def save(trainset):
    if not os.path.exists(TRANSLATION_PATH):
        os.mkdir(TRANSLATION_PATH)

    with open(os.path.join(TRANSLATION_PATH, 'semeval.de'), 'w') as f:
        f.write('\n'.join(map(lambda x: x['source'], trainset)))

    with open(os.path.join(TRANSLATION_PATH, 'semeval.en'), 'w') as f:
        f.write('\n'.join(map(lambda x: x['target'], trainset)))

if __name__ == '__main__':
    print('Load corpus')
    trainset, devset = load.run()
    print('Preparing training questions for alignment')
    trainset = prepare_questions(trainset)
    print('Saving Parallel data')
    save(trainset)