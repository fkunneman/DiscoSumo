__author__='floriankunneman'

import random
import sys
import json

import question_relator
from main import GoeieVraag

outfile = sys.argv[1]
outfile_shallow = sys.argv[2]
topic_percentage = float(sys.argv[3]) # topics to select
ncandidates = int(sys.argv[4]) # seed size
nrelate = int(sys.argv[5]) # total questions to query

# init
print('Initializing Similarity classifier')
sim_model = GoeieVraag(evaluation=False, w2v_dim=300)

print('Initializing Question relator')
qr = question_relator.QuestionRelator(sim_model)
qr.load_corpus()

# relate questions 
print('Relating questions')
output = []
output_shallow = []
sample = random.sample(range(len(sim_model.seeds)),nrelate)
sample_questions = [sim_model.seeds[i] for i in sample]
for question in sample_questions:
    output_question = [question['id'],question['text']]
    prominent_topics, candidates, candidates_ranked_sim, candidates_ranked_sim_filtered, candidates_first_topic_popularity, candidates_prominent_topics_popularity = qr.relate_question(question,topic_percentage,ncandidates)
    output_question.append([x['topic_text'] for x in prominent_topics])
    if len(candidates_first_topic_popularity) < 5:
        continue
    sim_all = ['SIM ALL']
    for x in candidates:
        sim_all.append({'id':x[0],'text':sim_model.seeds_text[x[0]],'score':x[4],'pop':x[3]})
    relevance = ['RELEVANCE']
    for x in candidates_ranked_sim:
        relevance.append({'id':x[0],'text':sim_model.seeds_text[x[0]],'score':x[4],'sim':x[5],'pop':x[3]})
    relevance_filtered = ['RELEVANCE FILTERED']
    for x in candidates_ranked_sim_filtered:
        relevance_filtered.append({'id':x[0],'text':sim_model.seeds_text[x[0]],'score':x[4],'sim':x[5],'pop':x[3]})
    first_topic = ['FIRST TOPIC']
    for x in candidates_first_topic_popularity:
        first_topic.append({'id':x[0],'text':sim_model.seeds_text[x[0]],'topic':x[-2],'score':x[-1],'pop':x[3]})
    all_topics = ['PROMINENT TOPICS']
    shallow = {'qid':question['id']}
    shallow_related = []
    for x in candidates_prominent_topics_popularity:
        all_topics.append({'id':x[0],'text':sim_model.seeds_text[x[0]],'topic':x[-2],'score':x[-1],'pop':x[3]})
        shallow_related.append({'qid':x[0],'questiontext':sim_model.seeds_text[x[0]]})
    shallow['related'] = shallow_related
    output.append([output_question,sim_all,relevance,relevance_filtered,first_topic,all_topics])
    output_shallow.append(shallow)

print('Done.')
with open(outfile,'w',encoding='utf=8') as out:
    json.dump(output,out)

with open(outfile_shallow,'w',encoding='utf-8') as out:
    json.dump(output_shallow,out)
