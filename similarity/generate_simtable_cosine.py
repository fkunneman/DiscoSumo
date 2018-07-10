
import numpy
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from quoll.classification_pipeline.functions import linewriter

import sys

questions_in = sys.argv[1] # binary, csr matrix with tfidf values
questions_txt_in = sys.argv[2]
layer1_in = sys.argv[3]
layer2_in = sys.argv[4]
sim_any_out = sys.argv[5]
sim_l1_out = sys.argv[6]
sim_l2_out = sys.argv[7]
anti_memory = int(sys.argv[8]) # to prevent memory error for big file, choose 1

print('loading data')
# load questions
loader = numpy.load(questions_in)
questions = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

# load questions txt
with open(questions_txt_in,'r',encoding='utf-8') as file_in:
    questions_txt = file_in.read().strip().split('\n')

# load layer 1 categories
with open(layer1_in,'r',encoding='utf-8') as file_in:
    layer1 = file_in.read().strip().split('\n')

# load layer 2 categories
with open(layer2_in,'r',encoding='utf-8') as file_in:
    layer2 = file_in.read().strip().split('\n')

# calculate sim table
if not anti_memory:
    print('calculating similarities')
    cs = cosine_similarity(questions)
    print('Done.')

# format output
print('Collecting top 3s')
output_any = []
output_l1 = []
output_l2 = []
for i in range(questions.shape[0]): # for each row
    print('Extracting topsim questions for question',i)
    if anti_memory:
        cs = cosine_similarity(questions[i,:],questions)
    # any question goes
    if anti_memory:
        row = [[j,val] for j,val in enumerate(list(cs[0,:]))]
    else:
        row = [[j,val] for j,val in enumerate(list(cs[i,:]))]
    sorted_row = sorted(row, key = lambda k : k[1],reverse = True)
    filtered_row = [x for x in sorted_row if x[1] > 0.25]
    if len(filtered_row) >= 5:
        top5 = sum([[questions_txt[j[0]],j[1]] for j in filtered_row[1:6]],[])
    else:
        top5 = sum([[questions_txt[j[0]],j[1]] for j in filtered_row],[])
    if anti_memory:
        with open(sim_any_out,'a',encoding='utf-8') as out:
            out.write(','.join([str(x) for x in [questions_txt[i]] + top5]) + '\n')
    else:
        output_any.append([questions_txt[i]] + top5)
    # layer 1
    if anti_memory:
        row = [[j,val] for j,val in enumerate(list(cs[0,:])) if layer1[i] == layer1[j]]
    else:
        row = [[j,val] for j,val in enumerate(list(cs[i,:])) if layer1[i] == layer1[j]]
    sorted_row = sorted(row, key = lambda k : k[1],reverse = True)
    filtered_row = [x for x in sorted_row if x[1] > 0.25]
    if len(filtered_row) >= 5:
        top5 = sum([[questions_txt[j[0]],j[1]] for j in filtered_row[1:6]],[])
    else:
        top5 = sum([[questions_txt[j[0]],j[1]] for j in filtered_row],[])
    if anti_memory:
        with open(sim_l1_out,'a',encoding='utf-8') as out:
            out.write(','.join([str(x) for x in [questions_txt[i]] + top5]) + '\n')
    else:
        output_l1.append([questions_txt[i]] + top5)
    # layer 2
    if anti_memory:
        row = [[j,val] for j,val in enumerate(list(cs[0,:])) if layer2[i] == layer2[j]]
    else:
        row = [[j,val] for j,val in enumerate(list(cs[i,:])) if layer2[i] == layer2[j]]
    sorted_row = sorted(row, key = lambda k : k[1],reverse = True)
    filtered_row = [x for x in sorted_row if x[1] > 0.25]
    if len(filtered_row) >= 5:
        top5 = sum([[questions_txt[j[0]],j[1]] for j in filtered_row[1:6]],[])
    else:
        top5 = sum([[questions_txt[j[0]],j[1]] for j in filtered_row],[])
    if anti_memory:
        with open(sim_l2_out,'a',encoding='utf-8') as out:
            out.write(','.join([str(x) for x in [questions_txt[i]] + top5]) + '\n')
    else:
        output_l2.append([questions_txt[i]] + top5)

# write output
print('Writing output')
lw_any = linewriter.Linewriter(output_any)
lw_any.write_csv(sim_any_out)

lw_l1 = linewriter.Linewriter(output_l1)
lw_l1.write_csv(sim_l1_out)

lw_l2 = linewriter.Linewriter(output_l2)
lw_l2.write_csv(sim_l2_out)
