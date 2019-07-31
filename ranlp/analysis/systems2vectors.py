
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

train_in = sys.argv[1]
data_in = sys.argv[2]
outfile = sys.argv[3]
raw_data_out = sys.argv[4]
system_output = sys.argv[5:]

stop = set(stopwords.words('english'))

def vectorize_all(systems_predictions,datadict,vocab):
    systems_vector = defaultdict(list)
    raw_data = []
    # for each question
    for i,query in enumerate(sorted(systems_predictions[0][1].keys())):
        for system_predictions in systems_predictions:  
            targets_ranked = sorted(system_predictions[1][query],key = lambda k : k[1],reverse=True)
            predictions_quality = [x[-1] for x in targets_ranked]
            avp = average_precision(predictions_quality)
            systems_vector[system_predictions[0]].append(avp)
        to_rank = predictions_quality.count(1)
        for qt in systems_predictions[0][1][query]:
            d = datadict[qt[0]]
            qid = d[0]
            q = d[1]
            qsw = return_num_stopwords(q)
            qp = return_punctuation_stats(q)
            ql = return_capitalization_stats(q)
            qoov = return_oov(vocab,q)
            qt_id = d[2]
            qt_text = d[3]
            gs = qt[-1]
            qtsw = return_num_stopwords(qt_text)
            qtp = return_punctuation_stats(qt_text)
            qtl = return_capitalization_stats(qt_text)
            qtoov = return_oov(vocab,qt_text)
            matching = return_matching_words(q,qt_text)
            raw_data.append(['Q' + str(i+1),to_rank,qid,q,qsw,qp,ql,qoov,qt_id,qt_text,qtsw,qtp,qtl,qtoov,matching,gs])  

    return systems_vector, raw_data

def parse_predictionfile(predictionfile):    # read in predictions of system
    file_in = pickle.load(open(predictionfile, 'rb'))
    data = file_in['dev']
    question_predictions = defaultdict(list)
    for i,query_id in enumerate(sorted(data.keys())):
        for target_id in data[query_id].keys():
            score = data[query_id][target_id][0]
            gold_standard = data[query_id][target_id][1]
            question_predictions[query_id].append([target_id,float(score),gold_standard])
    return question_predictions

def average_precision(predictions):

    # compute the number of relevant docs
    # get a list of precisions in the range(0,th)
    avg_prec = 0
    precisions = []
    num_correct = 0
    for i in range(10):
        if predictions[i] == 1:
            num_correct += 1
            precisions.append(num_correct/(i+1))

    if precisions:
        avg_prec = sum(precisions)/len(precisions)
    else:
        avg_prec = 0

    return avg_prec


def return_matching_words(txt1,txt2):
    matching = list(set(txt1) & set(txt2))
    return len(matching),round(len(matching)/len(txt1),2),round(len(matching)/len(txt2),2)

def return_num_stopwords(txt):
    sw = [w for w in txt.split() if w in stop]
    return len(sw), round(len(sw)/len(txt.split()),2)

def return_punctuation_stats(txt):
    p = [w for w in txt.split() if re.search('[\W]',w)]
    return len(p), round(len(p)/len(txt.split()),2)

def return_capitalization_stats(txt):
    c = [w for w in txt if w.isupper()]
    return len(c), round(len(c)/len(txt),2)

def return_oov(vocab,txt):
    oov = len(txt.split()) - len(list(set(txt.split()) & vocab))
    return oov, round(oov/len(txt.split()),2) 

### MAIN ###

# load trainset
print('LOAD TRAINSET')
train_txt = []
with open(train_in,'r',encoding='utf-8') as file_in:
    traindata = json.loads(file_in.read().strip())
for question in traindata.keys():
    train_txt.append(traindata[question]['tokens'])
    for dup in traindata[question]['duplicates']:
        train_txt.append(dup['rel_question']['tokens'])
train_vocabulary = set(sum(train_txt,[]))

# load corpus
print('LOAD CORPUS')
ids_text = {}
with open(data_in,'r',encoding='utf-8') as file_in:
    data = json.loads(file_in.read().strip())
for question in data.keys():
    for dup in data[question]['duplicates']:
        ids_text[dup['rel_question']['id']] = [question,' '.join(data[question]['tokens']),dup['rel_question']['id'],' '.join(dup['rel_question']['tokens'])]

# parse predictions for both files
print('PARSE PREDICTIONS')
systems_predictions = []
for predictionfile in system_output:
    parts = predictionfile.split('/')[-1].split('.')
    system = '-'.join(parts)
    systems_predictions.append([system,parse_predictionfile(predictionfile)])

# compare systems
print('GENERATE VECTORS')
systems_vector, raw_data = vectorize_all(systems_predictions,ids_text, train_vocabulary)

# write output
lw = linewriter.Linewriter(raw_data)
lw.write_csv(raw_data_out)
with open(outfile,'w',encoding='utf-8') as out:
    for i,system_predictions in enumerate(systems_predictions):
        out.write(system_predictions[0] + '\t' + ','.join([str(x) for x in systems_vector[system_predictions[0]]]) + '\n')
