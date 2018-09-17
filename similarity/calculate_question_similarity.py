
import sys
import numpy
from scipy import sparse
import multiprocessing

from quoll.classification_pipeline.functions import linewriter

seeded_vectors_in = sys.argv[1]
unseeded_vectors_in = sys.argv[2]
similarities_out = sys.argv[3]
#seeded_txt_in = sys.argv[3]
#unseeded_txt_in = sys.argv[4]
#rankings_out = sys.argv[5]

# functions for parallelization
def make_chunks(indices,nc=20):
    i = 0
    chunks=[]
    size = int(len(indices)/nc)
    for j in range(nc-1):
        chunks.append(indices[i:(i+size)])
        i += size
    chunks.append(indices[i:])
    return chunks

def query_similarity(chunk,output,i,seeded,unseeded):
    for index in chunk:
        query_similarities = []
        query = seeded[index,:].toarray()
        for question in unseeded:
            query_similarities.append(numpy.sum(query * question.toarray()))
        output.put([index,query_similarities])
    print('Chunk',i,'done.')

# read seeded vectors
loader = numpy.load(seeded_vectors_in)
seeded_csr = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
print('SHAPE SEEDED',seeded_csr.shape)

# read unseeded
loader = numpy.load(unseeded_vectors_in)
unseeded_csr = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
print('SHAPE UNSEEDED',unseeded_csr.shape)

# multiply values
print('Making chunks')
chunks = make_chunks(list(range(seeded_csr.shape[0])))

print('Making calculations in parallel')
q = multiprocessing.Queue()
query_similarities = []
for i in range(len(chunks)):
    p = multiprocessing.Process(target=query_similarity,args=[chunks[i],q,i,seeded_csr,unseeded_csr])
    p.start()

index_query_similarities = []
while True:
    l = q.get()
    index_query_similarities.append(l)
    print(len(index_query_similarities),'of',seeded_csr.shape[0])
    if len(index_query_similarities) == seeded_csr.shape[0]:
        break

print('Reordering matrix')
index_query_similarities_sorted = sorted(index_query_similarities,key = lambda k : k[0])
print('First ten',[x[0] for x in index_query_similarities_sorted[:10]])
queries_similarities = [line[1] for line in index_query_similarities_sorted]
queries_similarities_csr = sparse.csr_matrix(queries_similarities)
print('Done. New shape:',queries_similarities_csr.shape,'Now writing to output')
numpy.savez(similarities_out, data=queries_similarities_csr.data, indices=queries_similarities_csr.indices, indptr=queries_similarities_csr.indptr, shape=queries_similarities_csr.shape)

# rank sims
# print('Ranking similarities by query')
# queries_ranked = []
# for i,row in enumerate(queries_similarities):
#     print('Query',i)
#     row_indices = [[i,x] for i,x in enumerate(row)]
#     row_indices_sorted = sorted(row_indices,key = lambda k : k[1],reverse=True)
#     queries_ranked.extend([[i,ri[0],ri[1]] for ri in row_indices_sorted[:100]])
 
# print('Reading texts')
# # read seeded text
# with open(seeded_txt_in,'r',encoding='utf-8') as file_in:
#     seeded_text = file_in.read().strip().split('\n')
# print('Length seeded text',len(seeded_text))

# # read unseeded text
# with open(unseeded_txt_in,'r',encoding='utf-8') as file_in:
#     unseeded_text = file_in.read().strip().split('\n')
# print('Length unseeded text',len(unseeded_text))
    
# # prepare output
# print('Preparing output')
# queries_ranked_txt = []
# for pair in queries_ranked:
#     queries_ranked_txt.append([pair[0],seeded_text[pair[0]],pair[1],unseeded_text[pair[1]],pair[2]])
    
# # write output
# print('Writing output')
# lw = linewriter.Linewriter(queries_ranked_txt)
# lw.write_csv(rankings_out)



