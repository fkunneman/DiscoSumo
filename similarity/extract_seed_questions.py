
import sys
import json
import random

from quoll.classification_pipeline.functions import linewriter

questions_in = sys.argv[1]
seed_questions_out = sys.argv[2]
seed_meta_out = sys.argv[3]
seed_descriptions_out = sys.argv[4]
rest_questions_out = sys.argv[5]
rest_meta_out = sys.argv[6]
rest_descriptions_out = sys.argv[7]
categories_json = sys.argv[8]
seed_size = int(sys.argv[9])

# open parsed questions
with open(questions_in,'r',encoding='utf-8') as file_in:
    questions = json.loads(file_in.read().strip())

# read categories
with open(categories_json,'r',encoding='utf-8') as file_in:
    categories = json.loads(file_in.read().strip())
    
# set up dictionary based on category ids
catdict = {}
for cat in categories:
    catdict[cat['id']] = cat['category']
    
# draw sample
big_pool = [[i,q] for i,q in enumerate(questions) if q['answercount'] != '0']
pool = [x for x in big_pool if x[1]['bestanswer'] != '0']
print('num questions',len(questions),'size big pool',len(big_pool),'size pool',len(pool))
seed = random.sample(pool,seed_size)
seed_indices = [x[0] for x in seed]
not_seeded = [x for x in big_pool if x[0] not in seed_indices]

# prepare data to write
seed_meta = [['id','index','uid','datetime','num answers','best answer','starcount','subcategory']]
seed_instances = []
seed_descriptions = []
for q in seed:
    seed_instances.append(q[1]['questiontext'].replace('\n',' '))
    metaline = [q[1]['id'],q[0],q[1]['uid'],q[1]['entered'],q[1]['answercount'],q[1]['bestanswer'],q[1]['starcount'],catdict[q[1]['cid']]]
    seed_meta.append(metaline)
    seed_descriptions.append(q[1]['description'].replace('\n',' '))

unseeded_meta = [['id','index','uid','datetime','num answers','best answer','starcount','main category','subcategory']]
unseeded_instances = []
unseeded_descriptions = []
for q in not_seeded:
    unseeded_instances.append(q[1]['questiontext'].replace('\n',' '))
    metaline = [q[1]['id'],q[0],q[1]['uid'],q[1]['entered'],q[1]['answercount'],q[1]['bestanswer'],q[1]['starcount'],catdict[q[1]['cid']]]
    unseeded_meta.append(metaline)
    unseeded_descriptions.append(q[1]['description'].replace('\n',' '))

# write to files
with open(seed_questions_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(seed_instances))

with open(seed_descriptions_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(seed_descriptions))

# write to output
lw = linewriter.Linewriter(seed_meta)
lw.write_csv(seed_meta_out)

with open(rest_questions_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(unseeded_instances))

with open(rest_descriptions_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(unseeded_descriptions))

# write to output
lw = linewriter.Linewriter(unseeded_meta)
lw.write_csv(rest_meta_out)
    
    

