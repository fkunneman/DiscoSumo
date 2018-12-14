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
calcs_out = sys.argv[3]

stop = set(stopwords.words('english'))

def run_statistics(vocabulary,questions,cat):
    stopwords = [return_num_stopwords(question) for question in questions]
    punctuation = [return_punctuation_stats(question) for question in questions]
    capitalization = [return_capitalization_stats(question) for question in questions]
    oov = [return_oov(vocabulary,question) for question in questions]
    stopwords_stats = return_average(stopwords)
    stopwords_out = [cat + ' Avg. stopwords per question: ' + str(stopwords_stats[0]) + ' (' + str(stopwords_stats[1]) + ')', cat + ' Avg. percentage of stopwords per question: ' + str(stopwords_stats[2]) + ' (' + str(stopwords_stats[3]) + ')']
    punctuation_stats = return_average(punctuation)
    punctuation_out = [cat + ' Avg. punctuation per question: ' + str(punctuation_stats[0]) + ' (' + str(punctuation_stats[1]) + ')', cat + ' Avg. percentage of punctuation per question: ' + str(punctuation_stats[2]) + ' (' + str(punctuation_stats[3]) + ')']
    capitalization_stats = return_average(capitalization)
    capitalization_out = [cat + ' Avg. capitals per question: ' + str(capitalization_stats[0]) + ' (' + str(capitalization_stats[1]) + ')', cat + ' Avg. percentage of capitals per question: ' + str(capitalization_stats[2]) + ' (' + str(capitalization_stats[3]) + ')']
    oov_stats = return_average(oov)
    oov_out = [cat + ' Avg. OOV per question: ' + str(oov_stats[0]) + ' (' + str(oov_stats[1]) + ')', cat + ' Avg. percentage of OOV per question: ' + str(oov_stats[2]) + ' (' + str(oov_stats[3]) + ')',]
    type_token = return_type_token(questions)
    type_token_out = [cat + ' Type-token ratio: ' + str(type_token)]
    return sum([stopwords_out,punctuation_out,capitalization_out,oov_out,type_token_out],[])

def run_matching_statistics(pairs):
    matching = [return_matching_words(pair[0],pair[1]) for pair in pairs]
    matching_stats = return_average(matching,n=3)
    matching_out = ['Avg. number of matching words per question: ' + str(matching_stats[0]) + ' (' + str(matching_stats[1]) + ')','Avg. percentage of matching words in query: ' + str(matching_stats[2]) + ' (' + str(matching_stats[3]) + ')','Avg. percentage of matching words in target: ' + str(matching_stats[4]) + ' (' + str(matching_stats[5]) + ')']
    return matching_out

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

def return_average(stats,n=2):
    average = []
    for i in range(n):
        average.extend([numpy.mean([x[i] for x in stats]),numpy.std([x[0] for x in stats])])
    return(average)

def return_type_token(txtlist):
    all_words = sum([x.split() for x in txtlist],[])
    unique_words = list(set(all_words))
    return len(unique_words) / len(all_words)

def return_oov(vocab,txt):
    oov = len(txt.split()) - len(list(set(txt.split()) & vocab))
    return oov, round(oov/len(txt.split())) 

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
query_txt = []
target_txt = []
pairs = []
with open(data_in,'r',encoding='utf-8') as file_in:
    data = json.loads(file_in.read().strip())
for question in data.keys():
    query_txt.append(' '.join(data[question]['tokens']))
    for dup in data[question]['duplicates']:
        target_txt.append(' '.join(dup['rel_question']['tokens']))
        pairs.append([query_txt[-1],target_txt[-1]])
print('Done.')
print('Number of queries:',len(query_txt))
print('Number of targets:',len(target_txt))

# calculate statistics for queries
print('RUN STATISTICS')
query_statistics = run_statistics(train_vocabulary,query_txt,cat='QUERY')
target_statistics = run_statistics(train_vocabulary,target_txt,cat='TARGET')
combined_statistics = run_statistics(train_vocabulary,query_txt+target_txt,cat='COMBINED')
matching_statistics = run_matching_statistics(pairs)

# write to output
out_str = '\n'.join(query_statistics + target_statistics + combined_statistics + matching_statistics)
with open(calcs_out,'w',encoding='utf-8') as out:
    out.write(out_str)
