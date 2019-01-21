__author__='thiagocastroferreira'

import json

from main import GoeieVraag

TRAINING_DATA='/roaming/fkunnema/goeievraag/exp_similarity/ranked_questions_labeled_proc.json'

def map(ranking, n=10):
    map_ = 0.0
    n = len(list(ranking.keys()))
    for qid in ranking:
        result = sorted(ranking[qid], key=lambda x: x[1], reverse=True)[:n]

        precision = []
        for i, row in enumerate(result):
            size = i+1
            tp = [w for w in result[:i+1] if w[2] == 1]

            precision.append(float(tp) / size)

        map_ += sum(precision) / len(precision)

    return map_ / n

if __name__ == '__main__':
    goeie = GoeieVraag()

    # BM25
    bm25ranking, transranking, softranking = {}, {}, {}
    testdata = goeie.testdata
    for q1id in testdata:
        bm25ranking[q1id] = []
        for q2id in testdata[q1id]:
            score, label = testdata[q1id][q2id]['score'], testdata[q1id][q2id]['label']
            bm25ranking[q1id].append((q2id, score, label))

            q1, q2 = testdata[q1id][q2id]['q1'], testdata[q1id][q2id]['q2']
            q1, q1emb, q2, q2emb = goeie.preprocess(q1=q1, q2=q2)

            transcore = goeie.translate(q1, q1emb, q2, q2emb)
            transranking[q1id].append((q2id, transcore, label))

            softscore = goeie.softcos(q1, q1emb, q2, q2emb)
            softranking[q1id].append((q2id, softscore, label))

    print('BM25: Evaluation')
    print('MAP:', round(map(bm25ranking), 4))
    print(10 * '-')

    print('Translation: Evaluation')
    print('MAP:', round(map(transranking), 4))
    print(10 * '-')

    print('Softcosine: Evaluation')
    print('MAP:', round(map(softranking), 4))
    print(10 * '-')
