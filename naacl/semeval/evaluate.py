__author__='thiagocastroferreira'

import os

from semeval_bm25 import SemevalBM25
from semeval_translation import SemevalTranslation
from semeval_cosine import SemevalCosine, SemevalSoftCosine

EVALUATION_PATH='evaluation'

if __name__ == '__main__':
    # bm25
    bm25 = SemevalBM25()
    ranking = bm25.validate()

    path = os.path.join(EVALUATION_PATH, 'bm25.ranking')
    bm25.save(ranking, path)

    # align translation
    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector='alignments')
    ranking = translation.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.align.ranking')
    translation.save(ranking, path)

    # word2vec translation
    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector='word2vec')
    ranking = translation.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.word2vec.ranking')
    translation.save(ranking, path)

    # align translation
    translation = SemevalTranslation(alpha=0.7, sigma=0.3, vector='word2vec+elmo')
    ranking = translation.validate()

    path = os.path.join(EVALUATION_PATH, 'translation.word2vec_elmo.ranking')
    translation.save(ranking, path)

    # cosine
    cosine = SemevalCosine()
    ranking = cosine.validate()

    path = os.path.join(EVALUATION_PATH, 'cosine.ranking')
    cosine.save(ranking, path)

    # align cosine
    aligncosine = SemevalSoftCosine(vector='aligments')
    ranking = aligncosine.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.align.ranking')
    cosine.save(ranking, path)

    # word2vec cosine
    softcosine = SemevalSoftCosine(vector='word2vec')
    ranking = softcosine.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.word2vec.ranking')
    cosine.save(ranking, path)

    # word2vec + elmo cosine
    softcosine = SemevalSoftCosine(vector='word2vec+elmo')
    ranking = softcosine.validate()

    path = os.path.join(EVALUATION_PATH, 'softcosine.word2vec_elmo.ranking')
    cosine.save(ranking, path)