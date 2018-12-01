
import os
import sys
import json
from collections import defaultdict

from nltk.corpus import stopwords

predictions_system1_in = sys.argv[1]
predictions_system2_in = sys.argv[2]
prediction_type = sys.argv[3] # ranked or clf 
data_in = sys.argv[4] # the file with full data stored in json, to get the question txt
outdir = sys.argv[5]

stop = set(stopwords.words('english'))

def make_assessments(system_predictions):
    system_assessments = {}
    # for each question
    for query in sorted(system_predictions.keys()): 
        assessments = []
        # if ranking
        if prediction_type == 'ranked':
            # rank targets
            targets_ranked = sorted(system_predictions[query],key = lambda k : k[1],reverse=True)
            targets_ranked_numbered = [[i] + x for i,x in enumerate(targets_ranked)]
            num_true = [x[3] for x in targets_ranked_numbered].count('true')
            # for each target
            for target in targets_ranked_numbered:
                rank = target[0]
                gs = target[3]
                if rank < num_true and gs == 'true': # good call
                    target.extend(['true positive','-'])
                elif rank < num_true and gs == 'false':
                    diff = num_true-rank
                    print(diff)
                    target.extend(['false positive',str(diff)])
                elif rank > num_true and gs == 'true':
                    diff = rank-num_true
                    target.extend(['false negative',str(diff)])
                else:
                    target.extend(['false positive','-'])
                assessments.append(target)
        # if binary
        if prediction_type == 'binary':
            for target in system_predictions[query]:
                if target[1] >= 0 and target[2] == 'true':
                    target.append('true positive')
                elif target[1] >= 0 and target[2] == 'false':
                    target.append('false positive')
                elif target[1] < 0 and target[2] == 'true':
                    target.append('false negative')
                else:
                    target.append('true negative')
                assessments.append(target)
        system_assessments[query] = assessments
    return system_assessments

def parse_predictionfile(predictionfile):
    # read in predictions of system
    with open(predictionfile) as file_in:
        predictionlines = file_in.read().strip().split('\n')[1:]

    question_predictions = defaultdict(list)
    for prediction in predictionlines:
        tokens = prediction.split('\t')
        query_id = tokens[0]
        target_id = tokens[1].split('_')[1]
        score = tokens[3]
        gold_standard = tokens[4]
        question_predictions[query_id].append([target_id,float(score),gold_standard])
    return question_predictions 

def highlight_differences(assessments1,assessments2):
    pp = []
    pn = []
    np = []
    nn = []
    for query in sorted(assessments1.keys()):
        targets1 = assessments1[query]
        targets2 = assessments2[query]
        if prediction_type == 'ranked':
            p1 = [x for x in targets1 if x[-2][:4] == 'true']
            p2 = [x for x in targets2 if x[-2][:4] == 'true']
            n1 = [x for x in targets1 if x[-2][:5] == 'false']
            n2 = [x for x in targets2 if x[-2][:5] == 'false']
            p1_iddict =  {x[1]:x for x in p1}
            p2_iddict =  {x[1]:x for x in p2}
            n1_iddict =  {x[1]:x for x in n1}
            n2_iddict =  {x[1]:x for x in n2}
        if prediction_type == 'binary':
            p1 = [x for x in targets1 if x[-1][:4] == 'true']
            p2 = [x for x in targets2 if x[-1][:4] == 'true']
            n1 = [x for x in targets1 if x[-1][:5] == 'false']
            n2 = [x for x in targets2 if x[-1][:5] == 'false']
            p1_iddict =  {x[0]:x for x in p1}
            p2_iddict =  {x[0]:x for x in p2}
            n1_iddict =  {x[0]:x for x in n1}
            n2_iddict =  {x[0]:x for x in n2}
        p1_ids = set(list(p1_iddict.keys()))
        p2_ids = set(list(p2_iddict.keys()))
        n1_ids = set(list(n1_iddict.keys()))
        n2_ids = set(list(n2_iddict.keys()))
        positives = list(p1_ids & p2_ids)
        negatives = list(n1_ids & n2_ids)
        posnegs = list(p1_ids & n2_ids)
        negposs = list(n1_ids & p2_ids)
        for target_id in positives:
            pp.append([query,target_id] + p1_iddict[target_id] + p2_iddict[target_id])
        for target_id in negatives:
            nn.append([query,target_id] + n1_iddict[target_id] + n2_iddict[target_id])
        for target_id in posnegs:
            pn.append([query,target_id] + p1_iddict[target_id] + n2_iddict[target_id])
        for target_id in negposs:
            np.append([query,target_id] + n1_iddict[target_id] + p2_iddict[target_id])
    return pp,pn,np,nn

def return_matching_words(txt1,txt2):
    matching = list(set(txt1) & set(txt2))
    return len(matching),round(len(matching)/len(txt1),2),round(len(matching)/len(txt2),2)

def return_num_stopwords(txt):
    sw = [w for w in txt if w in stop]
    return len(sw), round(len(sw)/len(txt),2)

def write_out(outlines,qid_txt,outdir,outfilename):
    if prediction_type == 'ranked':
        outlines_txt = [['Query ID','Target ID','Query text','Target text','Gold standard','S1 Rank','S1 score','S1 assessment','S1 performance','S2 Rank','S2 score','S2 assessment','S2 performance','#words query','#words','#matching','%matching_query','%matching_target','#stopwords_query','%stopwords_query','#stopwords_target','%stopwords_target']]
    else:
        outlines_txt = [['Query ID','Target ID','Query text','Target text','Gold standard','S1 score','S1 assessment','S2 score','S2 assessment','#words query','#words','#matching','%matching_query','%matching_target','#stopwords_query','%stopwords_query','#stopwords_target','%stopwords_target']]
    for line in outlines:
        query_txt = qid_txt[line[0]]
        target_txt = qid_txt[line[0] + '_' + line[1]]
        num_matching, percent_matching_query, percent_matching_target = return_matching_words(query_txt,target_txt)
        num_stopwords_query, percent_stopwords_query = return_num_stopwords(query_txt)
        num_stopwords_target, percent_stopwords_target = return_num_stopwords(target_txt)
        outline_txt = [query_txt,target_txt] + line + [len(query_txt),len(target_txt),num_matching,percent_matching_query,percent_matching_target,num_stopwords_query,percent_stopwords_query,num_stopwords_target,percent_stopwords_target]
        if prediction_type == 'ranked':
            outlines_txt.append([outline_txt[i] for i in [2,3,0,1,7,4,6,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24]])
        elif prediction_type == 'binary':
            print(len(outline_txt))
            outlines_txt.append([outline_txt[i] for i in [2,3,0,1,6,5,7,9,11,12,13,14,15,16,17,18,19,20]])
    with open(outdir + '/' + outfilename,'w',encoding='utf-8') as out:
        out.write('\n'.join(['\t'.join([str(col) for col in line]) for line in outlines_txt]))

### MAIN ###

# load corpus
print('LOAD CORPUS')
qid_txt = {}
with open(data_in,'r',encoding='utf-8') as file_in:
    data = json.loads(file_in.read().strip())
for question in data.keys():
    qid_txt[data[question]['id']] = ' '.join(data[question]['tokens'])
    for dup in data[question]['duplicates']:
        if data[question]['id'] == 'Q270':
            print(dup['rel_question'].keys())
            print(dup['rel_question']['ranking'])
            print(dup['rel_question']['tokens'])
            print(data[question]['id'],dup['rel_question']['id'],dup['rel_question']['relevance'],data[question]['subject'],dup['rel_question']['subject'])              
        qid_txt[dup['rel_question']['id']] = ' '.join(dup['rel_question']['tokens'])
         
# parse predictions for both files
print('PARSE PREDICTIONS')
question_predictions1 = parse_predictionfile(predictions_system1_in)
question_predictions2 = parse_predictionfile(predictions_system2_in)

# assess predictions
print('ASSESS PREDICTIONS')
system_assessments1 = make_assessments(question_predictions1)
system_assessments2 = make_assessments(question_predictions2)

# find differences
print('FIND DIFFERENCES')
positive1_positive2, positive1_negative2, negative1_positive2, negative1_negative2 = highlight_differences(system_assessments1,system_assessments2)  

# write output
print('WRITE OUTPUT')
if not os.path.exists(outdir):
    os.mkdir(outdir)
write_out(positive1_positive2,qid_txt,outdir,'positive_positive.txt')
write_out(positive1_negative2,qid_txt,outdir,'positive_negative.txt')
write_out(negative1_positive2,qid_txt,outdir,'negative_positive.txt')
write_out(negative1_negative2,qid_txt,outdir,'negative_negative.txt')