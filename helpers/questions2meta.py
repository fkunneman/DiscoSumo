
import sys
import json

from quoll.classification_pipeline.functions import linewriter

questions_json = sys.argv[1]
categories_json = sys.argv[2]
instances_out = sys.argv[3]
meta_out = sys.argv[4]
descriptions_out = sys.argv[5]

# read questions
with open(questions_json,'r',encoding='utf-8') as file_in:
    questions = json.loads(file_in.read().strip())

# read categories
with open(categories_json,'r',encoding='utf-8') as file_in:
    categories = json.loads(file_in.read().strip())

# set up dictionary based on category ids
catdict = {}
for cat in categories:
    catdict[cat['id']] = cat['category']

# extract data from questions; add same index categories
instances = []
meta = [['id','uid','datetime','num answers','best answer','starcount','main category','subcategory']]
descriptions = []
for q in questions:
    instances.append(q['questiontext'].replace('\n',' '))
    metaline = [q['id'],q['uid'],q['entered'],q['answercount'],q['bestanswer'],q['starcount'],catdict[q['cid']]]
    meta.append(metaline)
    descriptions.append(q['description'].replace('\n',' '))

# write to output
lw = linewriter.Linewriter(meta)
lw.write_csv(meta_out)

with open(instances_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(instances))

with open(descriptions_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(descriptions))
