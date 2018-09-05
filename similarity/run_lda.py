
# based on the github repo from teffland/disclda, which was in turn forked from lda-project/lda
# script should be run when in the python2 DLDA virtualenv (due to mix-up with the regular lda)

import sys
import codecs
import numpy
from scipy import sparse

import lda
from quoll.classification_pipeline.functions import linewriter

vectors_in = sys.argv[1] # .npz file with sparse vectors of the questions
vocabulary_in = sys.argv[2] # .txt file with as many lines as their are columns in the vectors file
n_topics = int(sys.argv[3])
out_model = sys.argv[4]

# read vectors
print('Reading vectors')
loader = numpy.load(vectors_in)
vectors = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
print('Done. Vectorshape:',vectors.shape)

# read vocabulary
print('Reading vocabulary')
with codecs.open(vocabulary_in,'r',encoding='utf-8') as file_in:
    vocabulary = file_in.read().strip().split('\n')
print('Done. Vocabulary size:',len(vocabulary))

# train model
print('Training model')
model = lda.LDA(n_topics=n_topics, random_state=0, n_iter=1500)
model.fit(vectors)
print('DONE.')

# write model
print('Writing model')
topics_features = model.components_.tolist()
lw = linewriter.Linewriter(topics_features)
lw.write_csv(out_model)
