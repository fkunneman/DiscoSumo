
import os
import sys
import json
from collections import defaultdict
import pickle
import re
import numpy
from scipy import stats

from nltk.corpus import stopwords

data_in = sys.argv[1] # the file with full data stored in json, to get the question txt
traindata_in = sys.argv[2]
calcs_out = sys.argv[3]
evaltype = sys.argv[4]
plotdir = sys.argv[5]
inputfiles = sys.argv[6:]

stop = set(stopwords.words('english'))

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

def make_assessments(system_predictions):
    system_assessments = {}
    # for each question
    for query in sorted(system_predictions.keys()): 
        assessments = []
        targets_ranked = sorted(system_predictions[query],key = lambda k : k[1],reverse=True)
        for x in targets_ranked:
            x[1] = x[1]+1
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
            elif rank > num_true and gs == 1:
                diff = rank-num_true
                target.extend(['false negative',str(diff)])
            else:
                target.extend(['false positive','-'])
            assessments.append(target)
        system_assessments[query] = assessments
    return system_assessments

def add_info(predictions,vocab):
    for query in predictions.keys():
        query_txt = qid_txt[query]
        for target in predictions[query]:
            target_txt = qid_txt[target[1]]
            num_matching, percent_matching_target = return_matching_words(query_txt,target_txt)
            num_stopwords, percent_stopwords = return_num_stopwords(query_txt + target_txt)
            num_punctuation, percent_punctuation = return_punctuation_stats(query_txt + target_txt)
            num_capitals, percent_capitals = return_capitalization_stats(query_txt + target_txt)
            num_oov, percent_oov = return_oov(vocab,query_txt + target_txt)
            target.extend([num_matching,num_stopwords,num_punctuation,num_capitals,num_oov])

def calculate_correlations(vocab,predictions,outfile):
    #print('PRED',predictions)
    matching_out = [[x[6],x[2]] for x in predictions]
    with open(outfile + 'matchingplot.txt','w') as out:
        out.write('\n'.join([' '.join([str(x) for x in row]) for row in matching_out]))
    percent_matching_target_corr = stats.pearsonr([x[2] for x in predictions],[x[6] for x in predictions])
    stopwords_out = [[x[7],x[2]] for x in predictions]
    with open(outfile + 'stopwordplot.txt','w') as out:
        out.write('\n'.join([' '.join([str(x) for x in row]) for row in stopwords_out]))
    percent_stopwords_target_corr = stats.pearsonr([x[2] for x in predictions],[x[7] for x in predictions])
    punct_out = [[x[8],x[2]] for x in predictions]
    with open(outfile + 'punctplot.txt','w') as out:
        out.write('\n'.join([' '.join([str(x) for x in row]) for row in punct_out]))
    percent_punctuation_target_corr = stats.pearsonr([x[2] for x in predictions],[x[8] for x in predictions])
    caps_out = [[x[9],x[2]] for x in predictions]
    with open(outfile + 'capsplot.txt','w') as out:
        out.write('\n'.join([' '.join([str(x) for x in row]) for row in caps_out]))
    percent_capitalization_target_corr = stats.pearsonr([x[2] for x in predictions],[x[9] for x in predictions])
    oov_out = [[x[10],x[2]] for x in predictions]
    with open(outfile + 'oovplot.txt','w') as out:
        out.write('\n'.join([' '.join([str(x) for x in row]) for row in oov_out]))
    percent_oov_target_corr = stats.pearsonr([x[2] for x in predictions],[x[10] for x in predictions])
    return [
        str(round(percent_matching_target_corr[0],3)),
        str(round(percent_matching_target_corr[1],4)),
        str(round(percent_stopwords_target_corr[0],3)),
        str(round(percent_stopwords_target_corr[1],4)),
        str(round(percent_punctuation_target_corr[0],3)),
        str(round(percent_punctuation_target_corr[1],4)),
        str(round(percent_capitalization_target_corr[0],3)),
        str(round(percent_capitalization_target_corr[1],4)),
        str(round(percent_oov_target_corr[0],3)),
        str(round(percent_oov_target_corr[1],4)),
    ]

def return_matching_words(txt1,txt2):
    matching = list(set(txt1) & set(txt2))
    return len(matching),round(len(matching)/len(txt2),2)

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
    oov = len(txt) - len(list(set(txt) & vocab))

    return oov, round(oov/len(txt),2) 

### MAIN ###

# load trainset
print('LOAD TRAINSET')
train_txt = []
with open(traindata_in,'r',encoding='utf-8') as file_in:
    traindata = json.loads(file_in.read().strip())
for question in traindata.keys():
    train_txt.append(traindata[question]['tokens'])
    for dup in traindata[question]['duplicates']:
        train_txt.append(dup['rel_question']['tokens'])
train_vocabulary = set(sum(train_txt,[]))
print('Done.')
print('number of questions:',len(train_txt))
print('Vocabulary size',len(train_vocabulary))

# load corpus
print('LOAD CORPUS')
qid_txt = {}
with open(data_in,'r',encoding='utf-8') as file_in:
    data = json.loads(file_in.read().strip())
for question in data.keys():
    qid_txt[data[question]['id']] = ' '.join(data[question]['tokens'])
    for dup in data[question]['duplicates']:
        qid_txt[dup['rel_question']['id']] = ' '.join(dup['rel_question']['tokens'])

# for each file
output = [','.join([
    'System',
    'Percent matching target corr','Percent matching target p',
    'Percent stopwords target corr','Percent stopwords target p',
    'Percent punctuation target corr','Percent punctuation target p',
    'Percent capitalization target corr','Percent capitalization target p'
    ])]
for f in inputfiles:
    print(f)
    # parse predictions
    question_predictions = parse_predictionfile(f)

    # assess predictions
    system_assessments = make_assessments(question_predictions)

    # add information
    add_info(system_assessments,train_vocabulary)
    predictions = sum([x[1] for x in system_assessments.items()],[])

    # calculate correlations
    outfile = plotdir + f.split('/')[-1][:-4] + '_'
    correlations = [f.split('/')[-1]] + calculate_correlations(train_vocabulary,predictions,outfile)
    output.append(','.join(correlations))

# write output to file
with open(calcs_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(output))
