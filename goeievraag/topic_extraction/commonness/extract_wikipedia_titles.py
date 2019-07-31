
import sys
import json

from dte.functions import entity_extractor
from dte.classes import commonness

infile_raw = sys.argv[1]
infile_frog = sys.argv[2]
outfile = sys.argv[3]
commonness_txt = sys.argv[4]
commonness_cls = sys.argv[5]
commonness_corpus = sys.argv[6]
ngrams_file = sys.argv[7]

# set commonness object
cs = commonness.Commonness()
cs.set_classencoder(commonness_txt, commonness_cls, commonness_corpus)
ngram_files_loaded = []
cs.set_dmodel(ngrams_file)

# read in raw questions
with open(infile_raw,'r',encoding='utf-8') as file_in:
    questions_raw = file_in.read().strip().split('\n')

# read in questions
with open(infile_frog, 'r', encoding = 'utf-8') as file_in:
    questions = json.loads(file_in.read().strip())

# extract entities
questions_entities = []
for i,question in enumerate(questions):
    lemmas = []
    tokens = []
    poss = []
    raw = questions_raw[i]
    for sentence in question:
        lemmas.extend([x['lemma'].lower() for x in sentence])
        tokens.extend([x['text'] for x in sentence])
        poss.extend([x['pos'] for x in sentence])
    # lemmas.extend(list(set(em['normalizedTerms']) - set(tokens)))
    ee = entity_extractor.EntityExtractor()
    ee.set_commonness(cs)
    ee.extract_entities(lemmas)
    ee.filter_entities_threshold(0.01)
    output = [raw]
    output.append('--'.join(['|'.join([x[0],str(round(x[1],2))]) for x in ee.entities]))
    questions_entities.append('\t'.join(output))

# write to output
with open(outfile,'w',encoding='utf-8') as out:
    out.write('\n'.join(questions_entities)) 
