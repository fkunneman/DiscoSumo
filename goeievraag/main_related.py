__author__='thiagocastroferreira'

import random
import sys
import json

import question_relator
from main import GoeieVraag

outfile = sys.argv[1]
questions_topics = sys.argv[2]
questions_labeled = sys.argv[3]
w2v_model = sys.argv[4]
tfidf_model = sys.argv[5]
stopwords_file = sys.argv[6]
dictpath = sys.argv[7]
topic_percentage = float(sys.argv[8])
ncandidates = int(sys.argv[9])
filter_threshold = float(sys.argv[10])
diversity_threshold = float(sys.argv[11])
nrelate = int(sys.argv[12])

# init
print('Initializing similarity classifier')
sim_model = GoeieVraag(evaluation=False, w2v_dim=300)
print('Initializing question relator')
qr = question_relator.QuestionRelator(sim_model,w2v_model,tfidf_model,stopwords_file)

# load
print('Loading question relator')
qr.load_corpus(questions_topics,questions_labeled,dictpath)

# relate questions BM25
print('Relating questions')
output = []
sample = random.sample(range(len(qr.unlabeled_questions)),nrelate)
sample_questions = [qr.unlabeled_questions[i] for i in sample]
for question in sample_questions:
    output_question = {'qid':question.qid}
    
    prominent_topics, candidates, candidates_ranked, candidates_ranked_filtered, candidates_disposed, candidates_ranked_filtered_pop, candidates_ranked_filtered_pop_diversified, candidates_ranked_filtered_pop_remain = qr.relate_question(question,topic_percentage,ncandidates,filter_threshold,diversity_threshold)
    print('Top diverse related questions to',question.question.encode('utf-8'),':','!!!'.join(['=='.join([x[1].question] + [str(y) for y in x[2:]]) for x in candidates_ranked_filtered_pop_diversified]).encode('utf-8'))
    print('Remaining related questions to',question.question.encode('utf-8'),':','!!!'.join(['=='.join([x[1].question] + [str(y) for y in x[2:]]) for x in candidates_ranked_filtered_pop_remain]).encode('utf-8'))
    related = [q[1].qid for q in candidates_ranked_filtered_pop_diversified]
    additional = 5 - len(related)
    if additional > 0:
        related.extend([q[1].qid for q in candidates_ranked_filtered_pop_remain[:additional]])
    output_question['related'] = related
    output.append(output_question)

print('Done.')
with open(outfile,'w',encoding='utf-8') as out:
    json.dump(output,out)

