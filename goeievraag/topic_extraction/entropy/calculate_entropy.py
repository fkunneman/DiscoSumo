
import sys
import json
from math import log
import numpy
from collections import defaultdict

from sklearn import preprocessing

candidates_in = sys.argv[1]
meta_in = sys.argv[2]
cat_meta_in = sys.argv[3]
entropy_out = sys.argv[4]

#################################
### Functions ###################
#################################

def calculate_term_cat_probs(frequencies):
    total_frequency = sum(frequencies)
    probs = []
    for freq in frequencies:
        probs.append(freq/total_frequency)
    return probs

def calculate_entropy(probs):
    ent = 0
    for i in probs:
        ent += - i*log(i)
    return ent
    
#################################
### Main ########################
#################################

# read candidates
print('Reading candidates')
with open(candidates_in,'r',encoding='utf-8') as file_in:
    candidates = json.loads(file_in.read())
print('Done.',len(candidates),'candidates')
    
# read meta
print('Reading meta info')
with open(meta_in,'r',encoding='utf-8') as file_in:
    meta = json.loads(file_in.read())
print('Done.',len(meta),'meta instances')

# read meta
print('Reading category meta info')
with open(cat_meta_in,'r',encoding='utf-8') as file_in:
    category_meta = json.loads(file_in.read())

# index categories
print('Indexing categories')
catdict = {}
for cat in category_meta:
    cid = cat['id']
    parentid = cat['parentid']
    if parentid == '1':
        catdict[cid] = cid
    else:
        catdict[cid] = parentid
categories = sorted(list(set(list(catdict.values()))))

# generate candidate - frequencies graph
print('Generating candidate-frequencies graph')
candidate_frequencies_lemma = defaultdict(lambda : defaultdict(int))
for i,line in enumerate(candidates):
    category = categories.index(catdict[meta[i]['cid']])
    for candidate in line:
        candidate_lemma = [x['lemma'] for x in candidate]
        candidate_frequencies_lemma[' '.join(candidate_lemma).lower()][category] += 1

print('Done',len(candidate_frequencies_lemma.keys()),'lemma candidates')    

# calculate entropy for each candidate
print('Calculating entropy for each candidate')
candidates_entropy = []
for candidate in candidate_frequencies_lemma.keys():
    l = candidate_frequencies_lemma[candidate].values()
    if sum(l) > 2:
        probs = calculate_term_cat_probs(l)
        entropy = calculate_entropy(probs)
        candidates_entropy.append([candidate,entropy])

# scale entropy
print('Scaling and sorting by entropy')
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
entropy_values_scaled = [x[0] for x in min_max_scaler.fit_transform(numpy.array([[x[1]] for x in candidates_entropy]))]
candidates_entropy_scaled = []
for i,entity in enumerate(candidates_entropy):
    candidates_entropy_scaled.append([entity[0],entropy_values_scaled[i]])
        
# sort by specificity
candidates_entropy_scaled_sorted = sorted(candidates_entropy_scaled,key = lambda k : k[1])

# write output
print('Writing to output')
with open(entropy_out,'w',encoding='utf-8') as out:
    out.write('\n'.join([' '.join([str(x) for x in info]) for info in candidates_entropy_scaled_sorted]))
