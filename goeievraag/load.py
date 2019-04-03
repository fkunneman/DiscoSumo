__author__='thiagocastroferreira'

import os
import re
import spacy

from gensim import corpora

GOEIEVRAAG_PATH='/roaming/fkunnema/goeievraag'


def load_stopwords(path):
    with open(path) as f:
        stopwords = [word.lower().strip() for word in f.read().split()]
    return stopwords

def parse_corpus(fname, nlp, stopwords):
    vocabulary = []
    with open(fname) as f:
        doc = f.read()

    questions, questionsproc = [], []
    for q in doc.split('\n')[:-1]:
        question = list(map(lambda token: str(token).lower(), nlp(q)))
        vocabulary.extend(question)

        # lowercase
        question_proc = ' '.join(question)
        # remove special characters and punctuation
        question_proc = re.sub(r'[\W]+',' ', question_proc).strip()
        # remove stopwords
        question_proc = [word for word in question_proc.split() if word not in stopwords]

        if len(question_proc) > 0:
            questions.append(question)
            questionsproc.append(question_proc)

    return questions, questionsproc, vocabulary

def load_questions():
    nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
    stopwords = load_stopwords()

    path = os.path.join(GOEIEVRAAG_PATH, 'exp_similarity', 'train_questions.txt')
    trainset, trainproc, vocabulary = parse_corpus(path, nlp, stopwords)
    print('Number of train questions: ', str(len(trainproc)))

    path = os.path.join(GOEIEVRAAG_PATH, 'exp_similarity', 'seed_questions.txt')
    testset, testproc, _ = parse_corpus(path, nlp, stopwords)
    print('Number of test questions: ', str(len(testproc)))

    return trainset, testset, trainproc, testproc, vocabulary

def load_answers():
    nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
    stopwords = load_stopwords()

    path = os.path.join(GOEIEVRAAG_PATH, 'exp_similarity', 'train_answers.txt')
    trainset, trainproc, vocabulary = parse_corpus(path, nlp, stopwords)
    print('Number of train answers: ', str(len(trainproc)))

    path = os.path.join(GOEIEVRAAG_PATH, 'exp_similarity', 'seed_answers.txt')
    testset, testproc, _ = parse_corpus(path, nlp, stopwords)
    print('Number of test answers: ', str(len(testproc)))

    return trainset, testset, trainproc, testproc, vocabulary

def load_descriptions():
    nlp = spacy.load('nl', disable=['tagger', 'parser', 'ner'])
    stopwords = load_stopwords()

    path = os.path.join(GOEIEVRAAG_PATH, 'exp_similarity', 'train_descriptions.txt')
    trainset, trainproc, vocabulary = parse_corpus(path, nlp, stopwords)
    print('Number of train descriptions: ', str(len(trainproc)))

    path = os.path.join(GOEIEVRAAG_PATH, 'exp_similarity', 'seed_descriptions.txt')
    testset, testproc, _ = parse_corpus(path, nlp, stopwords)
    print('Number of test descriptions: ', str(len(testproc)))

    return trainset, testset, trainproc, testproc, vocabulary

def load():
    vocabulary = []
    trainquestions, testquestions, trainprocquestions, testprocquestions, voc = load_questions()
    vocabulary.extend(voc)

    trainanswers, testanswers, trainprocanswers, testprocanswers, voc = load_answers()
    vocabulary.extend(voc)

    traindescriptions, testdescriptions, trainprocdescriptions, testprocdescriptions, voc = load_answers()
    vocabulary.extend(voc)

    vocabulary.append('UNK')
    vocabulary.append('eos')
    vocabulary = list(set(vocabulary))

    id2voc = {}
    for i, trigram in enumerate(vocabulary):
        id2voc[i] = trigram

    voc2id = dict(map(lambda x: (x[1], x[0]), id2voc.items()))

    vocabulary = corpora.Dictionary(trainquestions + trainanswers + traindescriptions)
    return (
        trainquestions, trainprocquestions,
        testquestions, testprocquestions,
        trainanswers, trainprocanswers,
        testanswers, testprocanswers,
        traindescriptions, trainprocdescriptions,
        testdescriptions, testprocdescriptions,
        vocabulary, id2voc, voc2id
    )