
import sys
import json

in_frogged = sys.argv[1]
out_candidates = sys.argv[2]

# read frogged questions
print('Reading questions')
with open(in_frogged,'r',encoding='utf-8') as file_in:
    frogged = json.loads(file_in.read())

# extract candidates
print('Processing questions')
output = []
for question in frogged:
    candidates = []
    for sentence in question:
        adjective = False
        noun = False
        spec = False
        counter = 0
        for j,word in enumerate(sentence):
            if word['pos'] == 'WW':
                if adjective:
                    if counter == 1:
                        candidates.append(sentence[(j-1):j+1])
                    else:
                        candidates.append(sentence[(j-counter):j])
                        candidates.append([word])
                elif noun or spec:
                    candidates.append(sentence[(j-counter):j])
                else:
                    candidates.append([word])
                adjective = False
                noun = False
                spec = False
                counter = 0
            elif word['pos'] == 'SPEC':
                spec = True
                counter += 1
            elif word['pos'] == 'ADJ':
                adjective = True
                counter += 1
            elif word['pos'] == 'N':
                if noun or adjective:
                    candidates.append(sentence[(j-counter):j+1])
                else:
                    noun = True
                    counter += 1                
            else:
                if noun or adjective or spec:
                    candidates.append(sentence[(j-counter):j])
                adjective = False
                noun = False
                spec = False
                counter = 0
        if noun or adjective or spec:
            candidates.append(sentence[(j+1-counter):])
    output.append(candidates)
    
# write to output
with open(out_candidates,'w',encoding='utf-8') as out:
    json.dump(output,out)
