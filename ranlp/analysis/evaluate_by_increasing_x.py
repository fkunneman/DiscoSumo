
import os
import sys
import json
from collections import defaultdict
import pickle
import re
import numpy
from scipy import stats

from nltk.corpus import stopwords

train_in = sys.argv[1]
data_in = sys.argv[2] # the file with full data stored in json, to get the question txt
evaltype = sys.argv[3]
regression_out = sys.argv[4]
heatmap_outdir = sys.argv[5]
raw_outdir = sys.argv[6]
inputfiles = sys.argv[7:]

stop = set(stopwords.words('english'))

def calculate_ap(ranked_gold):
    precisions = []
    for i,retrieved in enumerate(ranked_gold):
    	if retrieved == 1:
            precisions.append(ranked_gold[:i+1].count(1)/(i+1))
    if ranked_gold.count(1) == 0:
    	return 0
    else:
    	ap = sum(precisions) / ranked_gold.count(1)
    return ap

def parse_predictionfile(predictionfile):    # read in predictions of system
    file_in = pickle.load(open(predictionfile, 'rb'))
    data = file_in[evaltype]
    question_predictions = {}
    for query_id in data.keys():
        question_predictions[query_id] = {}
        question_predictions[query_id]['targets'] = []
        for target_id in data[query_id].keys():
            score = data[query_id][target_id][0]
            gold_standard = data[query_id][target_id][1]
            question_predictions[query_id]['targets'].append([target_id,float(score),gold_standard])
        sorted_targets = sorted(question_predictions[query_id]['targets'],key = lambda k : k[1],reverse=True)
        question_predictions[query_id]['ap'] = calculate_ap([x[2] for x in sorted_targets])
    return question_predictions 

def add_info(predictions,txtdict,vocabulary):
    for query_id in predictions.keys():
    	txt = txtdict[query_id]
    	predictions[query_id]['stopwords_percent'] = return_num_stopwords(txt)
    	predictions[query_id]['punctuation_percent'] = return_punctuation_stats(txt)
    	predictions[query_id]['capitalization_percent'] = return_capitalization_stats(txt)
    	predictions[query_id]['oov_percent'] = return_oov(vocabulary,txt)
    	predictions[query_id]['matching_percent'] = return_avg_matching_words(predictions[query_id],txt,txtdict)

def return_avg_matching_words(qpred,qtxt,txtdict):
	matches = []
	for target in qpred['targets']:
		targettxt = txtdict[target[0]]
		matches.append(return_matching_words(qtxt,targettxt))
	return numpy.mean(matches)

def return_matching_words(txt1,txt2):
    matching = list(set(txt1) & set(txt2))
    return round(len(matching)/len(txt1),2)

def return_num_stopwords(txt):
    sw = [w for w in txt.split() if w in stop]
    return len(sw)/len(txt.split())

def return_punctuation_stats(txt):
    p = [w for w in txt.split() if re.search('[\W]',w)]
    return len(p)/len(txt.split())

def return_capitalization_stats(txt):
    c = [w for w in txt if w.isupper()]
    return len(c)/len(txt)

def return_oov(vocab,txt):
    oov = len(txt.split()) - len(list(set(txt.split()) & vocab))
    return oov/len(txt.split()) 


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
        qid_txt[dup['rel_question']['id']] = dup['rel_question']['tokens']

print('Done.')

# for each predictionfile
summary_out = []
heatmap = defaultdict(lambda : defaultdict(lambda : {}))
prepheaders = []
simheaders = []
for f in inputfiles:
    print(f)
    # parse predictions
    question_predictions = parse_predictionfile(f)

    # add information
    add_info(question_predictions,qid_txt,train_vocabulary)

    # run regression
    queries = question_predictions.keys()
    y = [question_predictions[q]['ap'] for q in queries]
    for datatype in ['stopwords_percent','punctuation_percent','capitalization_percent','oov_percent','matching_percent']:
        x = numpy.array([question_predictions[q][datatype] for q in queries])
        linreg = stats.linregress(x,y)
        summary_out.append(','.join([str(x) for x in [f,datatype,linreg.rvalue,linreg.pvalue,linreg.stderr]]))
        header = f.split('/')[-1].split('.')[0]
        if datatype in ['oov_percent','matching_percent']:
        	if header == 'bm25':
        		continue
        	column = f.split('/')[-1].split('.')[-1]
        	simheaders.append(header)
        else:	
        	if f.split('/')[-1].split('.')[-1].split('_')[0] == 'vector':
        		if (header == 'softcosine' and f.split('/')[-1].split('.')[-1].split('_')[1] != 'word2vec') or (header == 'translation' and f.split('/')[-1].split('.')[-1].split('_')[1] != 'alignments') or (header == 'kernel'):
        			continue
        		column = '.'.join(f.split('/')[-1].split('.')[1:-1])
        	else:
        		column = '.'.join(f.split('/')[-1].split('.')[1:])
        	prepheaders.append(header)
       	heatmap[datatype][column][header] = linreg.rvalue
        outfile = raw_outdir + '/' + f.split('/')[-1] + '__' + datatype + '.txt'
    outdata = [[str(x[i]),str(y[i])] for i in range(len(y))]
    with open(outfile,'w') as out:
    	out.write('\n'.join([' '.join(row) for row in outdata]))

for datatype in heatmap.keys():
	outfile = heatmap_outdir + '/' + datatype + '.txt'
	columns = sorted(heatmap[datatype].keys())
	outfile_columns = heatmap_outdir + '/' + datatype + '_columns.txt'
	if datatype in ['oov_percent','matching_percent']:
		headers = sorted(list(set(simheaders)))
	else:
		headers = sorted(list(set(prepheaders)))
	outfile_headers = heatmap_outdir + '/' + datatype + '_headers.txt'
	heatmap_out = []
	for column in columns:
		row = []
		for header in headers:
			try:
				row.append(str(heatmap[datatype][column][header]))
			except:
				row.append('0')
		heatmap_out.append(row)
	with open(outfile,'w') as out:
		out.write('\n'.join([','.join(row) for row in heatmap_out]))
	with open(outfile_columns,'w') as out:
		out.write('\n'.join(columns))
	with open(outfile_headers,'w') as out:
		out.write('\n'.join(headers))

# write to output
with open(regression_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(summary_out))
