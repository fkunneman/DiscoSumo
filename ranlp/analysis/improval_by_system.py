
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

outfile = sys.argv[1]
standard_system_output = sys.argv[2].split()
new_system_output = sys.argv[3].split()

stop = set(stopwords.words('english'))

def make_assessments(system1_predictions,system2_predictions):
    improved = 0
    unchanged = 0
    worsened = 0
    total = 0
    # for each question
    for query in sorted(system1_predictions.keys()): 
        targets_ranked1 = sorted(system1_predictions[query],key = lambda k : k[1],reverse=True)
        targets_ranked2 = sorted(system2_predictions[query],key = lambda k : k[1],reverse=True)
        indices1 = [i for i, x in enumerate(targets_ranked1) if x[-1] == 1]
        indices2 = [i for i, x in enumerate(targets_ranked2) if x[-1] == 1]
        total+=len(indices1)
        for i,x in enumerate(indices1):
            if indices2[i] == x:
                unchanged += 1
            elif indices2[i] < x:
                improved += 1
            else:
                worsened += 1

    percent_improved = round(improved / total,2)
    percent_unchanged = round(unchanged / total,2)
    percent_worsened = round(worsened / total,2)
    return [percent_improved, percent_unchanged, percent_worsened]

def parse_predictionfile(predictionfile):    # read in predictions of system
    file_in = pickle.load(open(predictionfile, 'rb'))
    data = file_in['dev']
    question_predictions = defaultdict(list)
    for query_id in data.keys():
        for target_id in data[query_id].keys():
            score = data[query_id][target_id][0]
            gold_standard = data[query_id][target_id][1]
            question_predictions[query_id].append([target_id,float(score),gold_standard])
    return question_predictions 

### MAIN ###

# parse predictions for both files
print('PARSE PREDICTIONS')
system_standard_predictions = {}
for predictionfile in standard_system_output:
    parts = predictionfile.split('/')[-1].split('.')
    system = parts[0] 
    system_standard_predictions[system] = parse_predictionfile(predictionfile)

system_new_predictions = {}
for predictionfile in new_system_output:
    parts = predictionfile.split('/')[-1].split('.')
    system = parts[0] 
    system_new_predictions[system] = parse_predictionfile(predictionfile)

# compare systems
print('COMPARE SYSTEMS')
system_stats = []
for system in system_standard_predictions.keys():
    print(system)
    system_stats.append([system] + make_assessments(system_standard_predictions[system],system_new_predictions[system]))

# write output
with open(outfile,'w',encoding='utf-8') as out:
    out.write('\n'.join(['\t'.join([str(x) for x in line]) for line in system_stats]))
