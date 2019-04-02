
import numpy
import pickle
import sys

from sklearn import preprocessing

questions_commonness_in = sys.argv[1]
scaler_out = sys.argv[2]

# read wiki
print('READING QUESTIONS')
info = {}
with open(questions_commonness_in,'r',encoding='utf-8') as file_in:
    lines = file_in.read().strip().split('\n')
    for line in lines:
        tokens = line.split('\t')
        question = tokens[0]
        entities = [entity.split('|') for entity in tokens[1].split('--')]
        for entity in entities:
            if len(entity) > 1:
                info[entity[0].lower()] = float(entity[1])

# prepare commonness values
wiki_entities = []
wiki_values = []
for it in info.items():
    wiki_entities.append(it[0])
    wiki_values.append([it[1]])

# scale values
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
print(wiki_values[-50:])
print(numpy.mean([x[0] for x in wiki_values]))
print(numpy.median([x[0] for x in wiki_values]))
print(min([x[0] for x in wiki_values]))

print([x[0] for x in min_max_scaler.fit_transform(numpy.array(wiki_values))][-50:])
quit()
min_max_scaler.fit(numpy.array(wiki_values))
print(min_max_scaler.transform(numpy.array([[0.1],[0.3],[0.75],[1.0],[2.5],[5.0]])))

# write to output
with open(scaler_out, 'wb') as fid:
    pickle.dump(min_max_scaler, fid)
