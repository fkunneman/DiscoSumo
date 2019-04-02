__author__='thiagocastroferreira'

import random
import sys

import question_relator

questions_topics = sys.argv[1]
questions_labeled = sys.argv[2]
w2v_model = sys.argv[3]
tfidf_model = sys.argv[4]
stopwords_file = sys.argv[5]
dictpath = sys.argv[6]
topic_percentage = float(sys.argv[7])
ncandidates = int(sys.argv[8])
filter_threshold = float(sys.argv[9])
diversity_threshold = float(sys.argv[10])
nrelate = int(sys.argv[11])

# init
print('Initializing question relator')
qr = question_relator.QuestionRelator(w2v_model,tfidf_model,stopwords_file)

# load
print('Loading question relator')
qr.load_corpus(questions_topics,questions_labeled,dictpath)

# relate questions BM25
print('Relating questions BM25')
sample = random.sample(range(len(qr.unlabeled_questions)),nrelate)
sample_questions = [qr.unlabeled_questions[i] for i in list(range(nrelate))]
for question in sample_questions:
    prominent_topics, candidates, candidates_ranked, candidates_ranked_filtered, candidates_disposed, candidates_ranked_filtered_pop, candidates_ranked_filtered_pop_diversified, candidates_ranked_filtered_pop_remain = qr.relate_question(question,topic_percentage,ncandidates,filter_threshold,diversity_threshold)
    print('Top diverse related questions to',question.question.encode('utf-8'),':','!!!'.join(['=='.join([x[1].question] + [str(y) for y in x[2:]]) for x in candidates_ranked_filtered_pop_diversified]).encode('utf-8'))
    print('Remaining related questions to',question.question.encode('utf-8'),':','!!!'.join(['=='.join([x[1].question] + [str(y) for y in x[2:]]) for x in candidates_ranked_filtered_pop_remain]).encode('utf-8'))

print('Done.')

# # relate questions SoftCosine
# print('Relating questions SoftCosine')
# sample = random.sample(qr.unlabeled_questions,nrelate)
# for q in sample:
#     qr.relate_softcosine(q)

