
import os
import sys
import json
from collections import defaultdict
import pickle
import re
import numpy
from scipy import stats

from nltk.corpus import stopwords
from quoll.classification_pipeline.functions import linewriter

prediction_type = sys.argv[1] # ranked or clf 
evaltype = sys.argv[2] # dev
traindata_in = sys.argv[3]
data_in = sys.argv[4] # the file with full data stored in json, to get the question txt
outdir = sys.argv[5]
predictions_in_setting1 = sys.argv[6].split() # give, for example, the predictions by all systems that had stopwords removed
predictions_in_setting2 = sys.argv[7].split() # give, for example, the predictions by all systems that did not have stopwords removed

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
            num_true = [x[3] for x in targets_ranked_numbered].count(1)
            # for each target
            for target in targets_ranked_numbered:
                rank = target[0]
                gs = target[3]
                if rank < num_true and gs == 1: # good call
                    target.extend(['true positive','-'])
                elif rank < num_true and gs == 0:
                    diff = num_true-rank
                    target.extend(['false positive',str(diff)])
                elif rank >= num_true and gs == 1:
                    diff = rank-num_true
                    target.extend(['false negative',str(diff)])
                else:
                    target.extend(['true negative','-'])
                assessments.append(target)
        # if binary
        if prediction_type == 'binary':
            for target in system_predictions[query]:
                if target[1] >= 0 and target[2] == 1:
                    target.append('true positive')
                elif target[1] >= 0 and target[2] == 0:
                    target.append('false positive')
                elif target[1] < 0 and target[2] == 1:
                    target.append('false negative')
                else:
                    target.append('true negative')
                assessments.append(target)
        system_assessments[query] = assessments
    return system_assessments

def parse_predictionfile(predictionfile):    # read in predictions of system
    file_in = pickle.load(open(predictionfile, 'rb'))
    data = file_in[evaltype]
    question_predictions = defaultdict(list)
    for query_id in data.keys():
        for target_id in data[query_id].keys():
            score = data[query_id][target_id][0]
            gold_standard = data[query_id][target_id][1]
            question_predictions[query_id].append([target_id,float(score),gold_standard])
    return question_predictions 

def compare_assessments(assessments1,assessments2,qtdict):
    # pp = []
    # pn = []
    # np = []
    # nn = []
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
            # pp.append([query,target_id] + p1_iddict[target_id] + p2_iddict[target_id])
            qtdict[target_id][4] += 1
        for target_id in negatives:
            # nn.append([query,target_id] + n1_iddict[target_id] + n2_iddict[target_id])
            qtdict[target_id][5] += 1
        for target_id in posnegs:
            # pn.append([query,target_id] + p1_iddict[target_id] + n2_iddict[target_id])
            qtdict[target_id][6] += 1
        for target_id in negposs:
            # np.append([query,target_id] + n1_iddict[target_id] + p2_iddict[target_id])
            qtdict[target_id][7] += 1
    # return pp,pn,np,nn,qtdict

def return_matching_words(txt1,txt2):
    matching = list(set(txt1) & set(txt2))
    return len(matching),round(len(matching)/len(txt1),2),round(len(matching)/len(txt2),2)

def return_num_stopwords(txt):
    sw = [w for w in txt if w in stop]
    return len(sw), round(len(sw)/len(txt),2)

def return_punctuation_stats(txt):
    p = [w for w in txt if re.search('[\W]',w)]
    return len(p), round(len(p)/len(txt),2)

def return_capitalization_stats(txt):
    c = [w for w in ' '.join(txt) if w.isupper()]
    return len(c), round(len(c)/len(txt),2)

def return_type_token(txtlist):
    all_words = sum([x.split() for x in txtlist],[])
    unique_words = list(set(all_words))
    return len(unique_words) / len(all_words)

def return_oov(vocab,txt):
    oov = len(txt.split()) - len(list(set(txt.split()) & vocab))
    return oov, round(oov/len(txt.split())) 

def summarize_stats(outlines):
    avg_score_sys1 = numpy.mean([x[6] for x in outlines[1:]])
    avg_score_sys2 = numpy.mean([x[10] for x in outlines[1:]])
    avg_sw_query = numpy.mean([x[19] for x in outlines[1:]])
    avg_sw_target = numpy.mean([x[21] for x in outlines[1:]])
    avg_sw = numpy.mean([x[19] for x in outlines[1:]] + [x[21] for x in outlines[1:]])
    avg_p_query = numpy.mean([x[23] for x in outlines[1:]])
    avg_p_target = numpy.mean([x[25] for x in outlines[1:]])
    avg_p = numpy.mean([x[23] for x in outlines[1:]] + [x[25] for x in outlines[1:]])
    avg_c_query = numpy.mean([x[27] for x in outlines[1:]])
    avg_c_target = numpy.mean([x[29] for x in outlines[1:]])
    avg_c = numpy.mean([x[27] for x in outlines[1:]] + [x[29] for x in outlines[1:]])
    return ['Average score system 1 -- ' + str(avg_score_sys1),'Average score system 2 -- ' + str(avg_score_sys2),'Average stopwords query -- ' + str(avg_sw_query),'Average stopwords target -- ' + str(avg_sw_target),'Average stopwords all -- ' + str(avg_sw),'Average punctuation query -- ' + str(avg_p_query),'Average punctuation target -- ' + str(avg_p_query),'Average punctuation all -- ' + str(avg_p),'Average capitalization query -- ' + str(avg_c_query),'Average capitalization target -- ' + str(avg_c_target),'Average capitalization all -- ' + str(avg_c)]

def write_out(outlines,qid_txt,vocab):
    if prediction_type == 'ranked':
        outlines_txt = [['Query ID','Target ID','Query text','Target text','Gold standard','S1 Rank','S1 score','S1 assessment','S1 performance','S2 Rank','S2 score','S2 assessment','S2 performance','#words query','#words','#matching','%matching_query','%matching_target','#stopwords_query','%stopwords_query','#stopwords_target','%stopwords_target','#punct_query','%punct_query','#punct_target','%punct_target','#caps_query','%caps_query','#caps_target','%caps_query']]
    else:
        outlines_txt = [['Query ID','Target ID','Query text','Target text','Gold standard','S1 score','S1 assessment','S2 score','S2 assessment','#words query','#words','#matching','%matching_query','%matching_target','#stopwords_query','%stopwords_query','#stopwords_target','%stopwords_target','#punct_query','%punct_query','#punct_target','%punct_target','#caps_query','%caps_query','#caps_target','%caps_query']]
    for line in outlines:
        query_txt = qid_txt[line[0]]
        target_txt = qid_txt[line[1]]
        num_matching, percent_matching_query, percent_matching_target = return_matching_words(query_txt,target_txt)
        num_stopwords_query, percent_stopwords_query = return_num_stopwords(query_txt)
        num_stopwords_target, percent_stopwords_target = return_num_stopwords(target_txt)
        num_punctuation_query, percent_punctuation_query = return_punctuation_stats(query_txt)
        num_punctuation_target, percent_punctuation_target = return_punctuation_stats(target_txt)
        num_capitals_query, percent_capitals_query = return_capitalization_stats(query_txt)
        num_capitals_target, percent_capitals_target = return_capitalization_stats(target_txt)
        num_oov_query, percent_oov_query = return_oov(vocab,query_txt)
        num_oov_target, percent_oov_target = return_oov(vocab,target_txt)
        outline_txt = [query_txt,target_txt] + line + [len(query_txt),len(target_txt),num_matching,percent_matching_query,percent_matching_target,num_stopwords_query,percent_stopwords_query,num_stopwords_target,percent_stopwords_target,num_punctuation_query,percent_punctuation_query,num_punctuation_target,percent_punctuation_target,num_capitals_query,percent_capitals_query,num_capitals_target,percent_capitals_target,num_oov_query, percent_oov_query,num_oov_target, percent_oov_target]
        if prediction_type == 'ranked':
            outlines_txt.append([outline_txt[i] for i in [2,3,0,1,7,4,6,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]])
        elif prediction_type == 'binary':
            outlines_txt.append([outline_txt[i] for i in [2,3,0,1,6,5,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]])
    return outlines_txt

def return_comparison(l1,l2):
    tpp, ppp = stats.ttest_ind(l1,l2)
    return [str(len(l1)) + ' - ' + str(len(l2)),str(round(numpy.mean(l1),2)) + ' (' + str(round(numpy.std(l1),2)) + ')',str(round(numpy.mean(l2),2)) + ' (' + str(round(numpy.std(l2),2)) + ')',str(round(tpp,2)),str(round(ppp,2))]

def compare_stats(pp,pn,np,nn,i):
    comparison_out = []
    if type(i) == list:
        stats_pp = [x[i[0]] for x in pp[1:]] + [x[i[1]] for x in pp[1:]]
        stats_pn = [x[i[0]] for x in pn[1:]] + [x[i[1]] for x in pn[1:]]
        stats_np = [x[i[0]] for x in np[1:]] + [x[i[1]] for x in np[1:]]
        stats_nn = [x[i[0]] for x in nn[1:]] + [x[i[1]] for x in nn[1:]]        
    else:
        stats_pp = [x[i] for x in pp[1:]]
        stats_pn = [x[i] for x in pn[1:]]
        stats_np = [x[i] for x in np[1:]]
        stats_nn = [x[i] for x in nn[1:]]
    comparison_out.append('\t'.join(['Pos-Pos'] + return_comparison(stats_pp,stats_pn + stats_np + stats_nn)))
    comparison_out.append('\t'.join(['Pos-Neg'] + return_comparison(stats_pn,stats_pp + stats_np + stats_nn)))
    comparison_out.append('\t'.join(['Neg-Pos'] + return_comparison(stats_np,stats_pn + stats_pp + stats_nn)))
    comparison_out.append('\t'.join(['Neg-Neg'] + return_comparison(stats_nn,stats_pn + stats_np + stats_pp)))
    return comparison_out

### MAIN ###

# load corpus
print('LOAD CORPUS')
assessment_counter = {}
with open(data_in,'r',encoding='utf-8') as file_in:
    data = json.loads(file_in.read().strip())
for question in data.keys():
    for dup in data[question]['duplicates']:
        if dup['rel_question']['id'] == 'Q268_R19':
            print('YES')
        assessment_counter[dup['rel_question']['id']] = [question,' '.join(data[question]['tokens']),dup['rel_question']['id'],' '.join(dup['rel_question']['tokens']),0,0,0,0]
         
# parse predictions for both files
print('PARSE PREDICTIONS')
system_assessments1 = {}
for predictionfile in predictions_in_setting1:
    parts = predictionfile.split('/')[-1].split('.')
    system = '.'.join([parts[0],parts[2],parts[3]]) 
    predictions = parse_predictionfile(predictionfile)
    assessment = make_assessments(predictions)
    system_assessments1[system] = assessment

system_assessments2 = {}
for predictionfile in predictions_in_setting2:
    parts = predictionfile.split('/')[-1].split('.')
    system = '.'.join([parts[0],parts[2],parts[3]]) 
    predictions = parse_predictionfile(predictionfile)
    assessment = make_assessments(predictions)
    system_assessments2[system] = assessment

# compare systems
print('COMPARE SYSTEMS')
for system in system_assessments1.keys():
    print(system)
    compare_assessments(system_assessments1[system],system_assessments2[system],assessment_counter)
#print(assessment_counter)

# rank examples
pstab_ranked = [x for x in sorted(assessment_counter.values(),key = lambda k : k[4],reverse=True) if x[4] != 0]
nstab_ranked = [x for x in sorted(assessment_counter.values(),key = lambda k : k[5],reverse=True) if x[5] != 0]
p_ranked = [x for x in sorted(assessment_counter.values(),key = lambda k : k[6],reverse=True) if x[6] != 0]
n_ranked = [x for x in sorted(assessment_counter.values(),key = lambda k : k[7],reverse=True) if x[7] != 0]

pstab_out = outdir + 'pstab.csv'
lw = linewriter.Linewriter(pstab_ranked)
lw.write_csv(pstab_out)

nstab_out = outdir + 'nstab.csv'
lw = linewriter.Linewriter(nstab_ranked)
lw.write_csv(nstab_out)

p_out = outdir + 'pos1.csv'
lw = linewriter.Linewriter(p_ranked)
lw.write_csv(p_out)

n_out = outdir + 'neg1.csv'
lw = linewriter.Linewriter(n_ranked)
lw.write_csv(n_out)

# # assess predictions
# print('ASSESS PREDICTIONS')
# system_assessments1 = make_assessments(question_predictions1)
# system_assessments2 = make_assessments(question_predictions2)

# # find differences
# print('FIND DIFFERENCES')
# positive1_positive2, positive1_negative2, negative1_positive2, negative1_negative2 = highlight_differences(system_assessments1,system_assessments2)  

# # write output
# print('WRITE OUTPUT')
# outdir_full = outdir + '/' + predictions_system1_in.split('/')[-1] + '---' + predictions_system2_in.split('/')[-1]
# if not os.path.exists(outdir_full):
#     os.mkdir(outdir_full)
# pp_outlines = write_out(positive1_positive2,qid_txt)
# pp_summarization = summarize_stats(pp_outlines)
# with open(outdir_full + '/positive_positive.txt','w',encoding='utf-8') as out:
#     out.write('\n'.join(pp_summarization) + '\n' + '\n'.join(['\t'.join([str(col) for col in line]) for line in pp_outlines]))
# pn_outlines = write_out(positive1_negative2,qid_txt)
# pn_summarization = summarize_stats(pn_outlines)
# with open(outdir_full + '/positive_negative.txt','w',encoding='utf-8') as out:
#     out.write('\n'.join(pn_summarization) + '\n' + '\n'.join(['\t'.join([str(col) for col in line]) for line in pn_outlines]))
# np_outlines = write_out(negative1_positive2,qid_txt)
# np_summarization = summarize_stats(np_outlines)
# with open(outdir_full + '/negative_positive.txt','w',encoding='utf-8') as out:
#     out.write('\n'.join(np_summarization) + '\n' + '\n'.join(['\t'.join([str(col) for col in line]) for line in np_outlines]))
# nn_outlines = write_out(negative1_negative2,qid_txt)
# nn_summarization = summarize_stats(nn_outlines)
# with open(outdir_full + '/negative_negative.txt','w',encoding='utf-8') as out:
#     out.write('\n'.join(nn_summarization) + '\n' + '\n'.join(['\t'.join([str(col) for col in line]) for line in nn_outlines]))

# all_targets1 = sum([x[1] for x in system_assessments1.items()],[])
# all_targets2 = sum([x[1] for x in system_assessments2.items()],[])
# comparisons = ['Average scores system 1 -- ' + str(round(numpy.mean([x[2] for x in all_targets1]),2)),'Average scores system 2 -- ' + str(round(numpy.mean([x[2] for x in all_targets2]),2))]
# for i,t in [[6,'Scores system1'],[10,'Scores system2'],[19,'Percent stopwords query'],[21,'Percent stopwords target'],[[19,21],'Percent stopwords_all'],[23,'Percent punctuation query'],[25,'Percent punctuation target'],[[23,25],'Percent punctuation all'],[27,'Percent capitals query'],[29,'Percent capitals target'],[[27,29],'Percent capitals all']]:
#     comparisons.extend([t] + compare_stats(pp_outlines,pn_outlines,np_outlines,nn_outlines,i))

# with open(outdir_full + '/summary.txt','w',encoding='utf-8') as out:
#     out.write('\n'.join(comparisons))
