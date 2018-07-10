
import sys

import sentence_semantics

categories = sys.argv[1]
questions = sys.argv[2]
stopwords = sys.argv[3]
model = sys.argv[4]
outdir = sys.argv[5]

# load categories
with open(categories,'r',encoding='utf-8') as file_in:
    cats = file_in.read().strip().split('\n')

# load questions
with open(questions,'r',encoding='utf-8') as file_in:
    qs = file_in.read().strip().split('\n')

# load stopwords
with open(stopwords,'r',encoding='utf-8') as file_in:
    sw = set(file_in.read().strip().split('\n'))
    
# remove stopwords from questions
ss = sentence_semantics.SentenceSemantics()
qs_nostopwords = []
for question in qs:
    qs_nostopwords.append(ss.remove_stopwords_sentence(question,sw))
    
# load model
ss.load_model(model)

# calculate similarities and write to file
for cat in cats:
    print(cat)
    related = ss.find_related_word(cat.lower().split())
    print('Related',', '.join([x[0] for x in related]).encode('utf-8'))
    continue
    sims = []
    for question in qs:
        sim_score, sim_output = ss.sentence_similarity_best_word(cat.lower().split(),question.lower().split())
        sims.append([question,sim_score,sim_output])
    sims_sorted = sorted(sims,key = lambda k : k[1],reverse = True)[:100]
    outfile = outdir + cat + '.txt'
    with open(outfile,'w',encoding='utf-8') as out:
        out.write('\n'.join(['\t'.join([str(x[0]),str(x[1]),' || '.join([' - '.join(y) for y in x[2]])]) for x in sims_sorted]))

