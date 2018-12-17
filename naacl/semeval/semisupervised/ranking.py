__author_='thiagocastroferreira'

import sys
sys.path.append('../')
import paths
import os

from gensim.summarization import bm25
from multiprocessing import Pool

SEMI_PATH=paths.SEMI_PATH

def load():
    with open(os.path.join(SEMI_PATH, 'index.txt')) as f:
        indexes = f.read().split('\n')

    with open(os.path.join(SEMI_PATH, 'question.proc.txt')) as f:
        questions_proc = [text.split() for text in f.read().split('\n')]

    return indexes, questions_proc

def init_bm25(corpus):
    model = bm25.BM25(corpus)
    average_idf = sum(map(lambda k: float(model.idf[k]), model.idf.keys())) / len(model.idf.keys())
    return model, average_idf


def retrieve(thread_id, corpus, mbm25, average_idf, n=30):
    result = {}
    for i, query in enumerate(corpus):
        p = round(float(i) / len(corpus), 3)
        if i % 100 == 0:
            print('Thread id: ', thread_id, 'Progress: ', p, i, sep='\t', end='\n')
        scores = mbm25.get_scores(query, average_idf)

        questions = [(j, scores[j]) for j in range(len(scores))]
        questions = sorted(questions, key=lambda x: x[1], reverse=True)[:n]
        result[i] = questions
    return result


def save(result):
    with open(os.path.join(SEMI_PATH, 'bm25_result.txt'), 'w') as f:
        for qid in result:
            row = [qid]
            for question in result[qid]:
                row.append('-'.join([str(question[0]), str(question[1])]))
            row.append('\n')
            f.write(' '.join(row))


def run(corpus, mbm25, average_idf, n):
    THREADS = 10
    n = int(len(corpus) / THREADS)
    chunks = [corpus[i:i+n] for i in range(0, len(corpus), n)]

    pool = Pool(processes=len(chunks))

    processes = []
    for i, chunk in enumerate(chunks):
        print('Process id: ', i+1, 'Doc length:', len(chunk))
        processes.append(pool.apply_async(retrieve, [i + 1, corpus, mbm25, average_idf, n]))

    ranking = {}
    for process in processes:
        result = process.get()
        ranking.update(result)

    pool.close()
    pool.join()

    return ranking

if __name__ == '__main__':
    print('Loading...')
    indexes, corpus = load()
    print('Initializing BM25...')
    model, avg_idf = init_bm25(corpus)
    print('Retrieving...\n')
    ranking = run(corpus, model, avg_idf, 30)
    print('Saving...\n')
    save(ranking)