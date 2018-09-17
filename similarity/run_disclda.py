
# based on the github repo from teffland/disclda, which was in turn forked from lda-project/lda
# script should be run when in the python2 DLDA virtualenv (due to mix-up with the regular lda)

import sys
import codecs
import numpy
from scipy import sparse
import csv

from disclda import dlda
from quoll.classification_pipeline.functions import linewriter

vectors_in = sys.argv[1] # .npz file with sparse vectors of the questions
vocabulary_in = sys.argv[2] # .txt file with as many lines as their are columns in the vectors file
meta_in = sys.argv[3] # .txt file with the category per question (as many as their are rows in the vectors file)
n_topics = int(sys.argv[4])
out_model = sys.argv[5]

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

# read categories
print('Reading categories')
with codecs.open(meta_in,'r',encoding='utf-8') as file_in:
    categories = [line.split(',')[-1] for line in file_in.read().strip().split('\n')[1:]]
print('Done. Num categories:',len(categories),'First 10:',categories[:10])
    
# convert categories to integers
print('Converting categories')
category_encoder = {}
category_decoder = {}
unique_categories = list(set(categories))
for i,cat in enumerate(unique_categories):
    category_encoder[cat] = i
    category_decoder[i] = cat
categories_encoded = numpy.array([category_encoder[cat] for cat in categories])
print('Done. Result:',categories_encoded,categories_encoded.shape)

# train model
print('Training model')
model = dlda.DiscLDA(n_topics=n_topics, random_state=0, n_iter=1500)
model.fit(vectors,categories_encoded)
print('DONE.')

# write model
print('Writing model')
topics_features = model.components_.tolist()
with open(out_model, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for line in topics_features:
        writer.writerow(line)
