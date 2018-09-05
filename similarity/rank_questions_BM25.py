
import sys

from gensim.summarization import bm25
from gensim.corpora import Dictionary

from quoll.classification_pipeline.functions import linewriter

seeded_questions_in = sys.argv[1]
seeded_questions_raw_in = sys.argv[2]
unseeded_questions_in = sys.argv[3]
unseeded_questions_raw_in = sys.argv[4]
stopwords_in = sys.argv[5]
ranked_out = sys.argv[6]
train_out = sys.argv[7]

# read files
print('Reading files')

with open(seeded_questions_in,'r',encoding='utf-8') as file_in:
    seeded_questions = file_in.read().strip().split('\n')

with open(seeded_questions_raw_in,'r',encoding='utf-8') as file_in:
    seeded_questions_raw = file_in.read().strip().split('\n')
    
with open(unseeded_questions_in,'r',encoding='utf-8') as file_in:
    unseeded_questions = file_in.read().strip().split('\n')

with open(unseeded_questions_raw_in,'r',encoding='utf-8') as file_in:
    unseeded_questions_raw = file_in.read().strip().split('\n')

with open(stopwords_in,'r',encoding='utf-8') as file_in:
    stopwords = file_in.read().strip().split('\n')
    
# set corpus
print('Setting corpus')
texts = [x.split() for x in seeded_questions] + [x.split() for x in unseeded_questions]
texts_no_stopwords = [list(set(words) - set(stopwords)) for words in texts]
dct = Dictionary(texts)  # initialize a Dictionary
corpus = [dct.doc2bow(text) for text in texts]
train = list(range(len(unseeded_questions)))

# set bm25 model
print('Initializing bm25 model')
model = bm25.BM25(corpus)

# get average idf
print('Calculating average idf')
average_idf = sum(map(lambda k: float(model.idf[k]), model.idf.keys())) / len(model.idf.keys())

# get top 50 ranked questions for each of the seed questions
print('Ranking questions')
output = [['seed question','unseeded question','bm25 score']]
for index,question in enumerate(seeded_questions):
    selection = []
    print('Question',index,'of',len(seeded_questions),'questions')
    scores = model.get_scores(dct.doc2bow(texts[index]), average_idf)
    scores_numbers = [[i,score] for i,score in enumerate(scores)]
    scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
    print('First index:',scores_numbers_ranked[0][0])
    c = 0
    while len(selection) <= 10:
        sn = scores_numbers_ranked[c]
        if sn[0] > len(seeded_questions):
            selection.append([seeded_questions_raw[index],unseeded_questions_raw[sn[0] - len(seeded_questions)],sn[1]])
            train.remove(sn[0] - len(seeded_questions))
        c += 1
    output.extend(selection)

# extract train questions
train_questions = [unseeded_questions[i] for i in train]
    
# write output
lw = linewriter.Linewriter(output)
lw.write_csv(ranked_out)

with open(train_out,'w',encoding='utf-8') as out:
    out.write('\n'.join(train_questions))
