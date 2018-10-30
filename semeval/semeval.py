__author__='thiagocastroferreira'

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': 'localhost', 'user': 'tcastrof'}
logger = logging.getLogger('tcpserver')

import features
import os
import utils
from translation import *
from semeval_svm import SemevalModel

TRANSLATION_PATH='translation/model/lex.f2e'
EVALUATION_PATH='results'
GOLD_PATH='/home/tcastrof/Question/semeval/evaluation/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy'

DATA_PATH='data'

class SemevalTranslation(SemevalModel):
    def __init__(self):
        SemevalModel.__init__(self)

    def train(self):
        pass

    def validate(self):
        logging.info('Validating', extra=d)
        simplelm, simpletrm, simpletrlm = {}, {}, {}
        lm, trm, trlm = {}, {}, {}
        for j, q1id in enumerate(self.devset):
            simplelm[q1id] = []
            simpletrm[q1id] = []
            simpletrlm[q1id] = []
            lm[q1id] = []
            trm[q1id] = []
            trlm[q1id] = []
            percentage = round(float(j+1) / len(self.devset), 2)
            print('Progress: ', percentage, j+1, sep='\t', end='\r')

            query = self.devset[q1id]
            q1 = query['tokens_proc']
            elmo_emb1 = self.develmo.get(str(self.devidx[q1id]))
            w2v_emb = features.encode(q1, self.word2vec)
            q1emb = [np.concatenate([w2v_emb[i], elmo_emb1[i]]) for i in range(len(w2v_emb))]

            duplicates = query['duplicates']
            for duplicate in duplicates:
                rel_question = duplicate['rel_question']
                q2id = rel_question['id']

                q2 = rel_question['tokens_proc']
                elmo_emb2 = self.develmo.get(str(self.devidx[q2id]))
                w2v_emb = features.encode(q2, self.word2vec)
                q2emb = [np.concatenate([w2v_emb[i], elmo_emb2[i]]) for i in range(len(w2v_emb))]

                slmprob, strmprob, strlmprob, _ = self.translation.score(q1, q2)
                lmprob, trmprob, trlmprob, _ = self.translation.score_embeddings(q1, q1emb, q2, q2emb)
                real_label = 0
                if rel_question['relevance'] != 'Irrelevant':
                    real_label = 1
                simplelm[q1id].append((real_label, slmprob, q2id))
                simpletrm[q1id].append((real_label, strmprob, q2id))
                simpletrlm[q1id].append((real_label, strlmprob, q2id))
                lm[q1id].append((real_label, lmprob, q2id))
                trm[q1id].append((real_label, trmprob, q2id))
                trlm[q1id].append((real_label, trlmprob, q2id))

        with open('data/translationranking.txt', 'w') as f:
            for q1id in trlm:
                for row in trlm[q1id]:
                    label = 'false'
                    if row[0] == 1:
                        label = 'true'
                    f.write('\t'.join([str(q1id), str(row[2]), str(0), str(row[1]), label, '\n']))

        logging.info('Finishing to validate.', extra=d)
        return simplelm, simpletrm, simpletrlm, lm, trm, trlm

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # Softcosine
    semeval = SemevalTranslation()

    simplelm, simpletrm, simpletrlm, lm, trm, trlm = semeval.validate()

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, simplelm)
    print('Evaluation simplelm')
    print('MAP baseline: ', map_baseline)
    print('MAP model: ', map_model)
    print(10 * '-')

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, simpletrm)
    print('Evaluation simpletrm')
    print('MAP model: ', map_model)
    print(10 * '-')

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, simpletrlm)
    print('Evaluation simpletrlm')
    print('MAP model: ', map_model)
    print(10 * '-')

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, lm)
    print('Evaluation lm')
    print('MAP model: ', map_model)
    print(10 * '-')

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, trm)
    print('Evaluation trm')
    print('MAP model: ', map_model)
    print(10 * '-')

    devgold = utils.prepare_gold(GOLD_PATH)
    map_baseline, map_model = utils.evaluate(devgold, trlm)
    print('Evaluation trlm')
    print('MAP model: ', map_model)
    print(10 * '-')