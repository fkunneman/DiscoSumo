
import sys
import json
from collections import defaultdict

from quoll.classification_pipeline.functions import linewriter

answers_parsed = sys.argv[1]
outdir = sys.argv[2]

# open answers
with open(answers_parsed,'r',encoding='utf-8') as file_in:
    answers = json.loads(file_in.read().strip())

# initialize dict
question_answers = defaultdict(list)
for answer in answers:
    question_answers[answer['qid']].append(answer)

# write files
for qid in question_answers.keys():
    print(qid)
    answers = question_answers[qid]
    meta = [['id','uid','datetime','thumbsdown','thumbsup','bestanswer']]
    for answer in answers:
        meta.append([answer['id'],answer['uid'],answer['entered'],answer['thumbsdowncount'],answer['thumbsupcount'],answer['bestanswer']])
    outfile_meta = outdir + 'answermeta_' + str(qid) + '.txt'
    lw = linewriter.Linewriter(meta)
    lw.write_csv(outfile_meta)
    outfile_txt = outdir + 'answertext_' + str(qid) + '.txt'
    with open(outfile_txt,'w',encoding='utf-8') as out:
        out.write('\n'.join([a['answertext'].replace('\n',' ') for a in answers]))
