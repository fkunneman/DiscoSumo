from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import string

import json
import colibricore

# exclude = set(string.punctuation)
exclude = set([".",",",":",")","(","\""])

tmp = sys.argv[1]
outdir = sys.argv[2]
infiles = sys.argv[3:]

classfile = tmp + "_page.colibri.cls"

textfile = tmp + "_page.txt"
corpusfile = tmp + "_page.colibri.dat"

print("Writing all texts to temporary file" ,file=sys.stderr)
with open(textfile,'w',encoding='utf-8') as g:
    for i, infile in enumerate(infiles):
        with open(infile,encoding = "utf-8") as f:
            for l in f.readlines():
                js = json.loads(l)
                text = js["text"].lower()
                #text = ''.join(ch for ch in text if ch not in exclude)
                text = text.replace(',',' ,')
                text = text.replace('.',' .')
                text = text.replace(':',' :')
                text = text.replace('(','')
                text = text.replace(')','')
                text = text.replace('"','')
                g.write(text.strip() + "\n")

print("Building class encoder",file=sys.stderr)
classencoder = colibricore.ClassEncoder()
classencoder.build(textfile)
classencoder.save(classfile)

print("Encoding corpus data",file=sys.stderr)
classencoder.encodefile(textfile, corpusfile)

print("Loading class decoder",file=sys.stderr)
classdecoder = colibricore.ClassDecoder(classfile)

anchormodel = colibricore.UnindexedPatternModel()
print("Counting anchors",file=sys.stderr)

for i, infile in enumerate(infiles):
    with open(infile,encoding = "utf-8") as f:
        for l in f.readlines():
            js = json.loads(l)
            text = js["text"].lower()
            # text = ''.join(ch for ch in text if ch not in exclude)
            text = text.replace(',',' ,')
            text = text.replace('.',' .')
            text = text.replace(':',' :')
            text = text.replace('(','')
            text = text.replace(')','')
            text = text.replace('"','')
            #print('0000000000000000000000000',text)
            anchors = js["annotations"]
            #surface1 = [x["surface_form"].strip().lower() for x in anchors]
            surface = [''.join(ch for ch in x["surface_form"].strip().lower() if ch not in exclude) for x in anchors]
            #print('11111111111111111111111',surface1,'2222222222222222222',surface)
            for ngram in surface:
                if ngram:
                    pattern = classencoder.buildpattern(ngram)
                    if pattern.unknown():
                        print("WARNING: Anchor has unknown part " +  ngram + ", skipping... (" + pattern.tostring(classdecoder) + ")",file=sys.stderr)
                    else:
                        if len(pattern) <= 5:
                            anchormodel.add(pattern) #(will count +1  if already exists)

print("Anchors found: ", len(anchormodel),file=sys.stderr)

print("Counting n-grams, constrained by anchors",file=sys.stderr)
options = colibricore.PatternModelOptions(mintokens=1, maxlength=5)
patternmodel = colibricore.UnindexedPatternModel()
patternmodel.train(corpusfile, options, anchormodel) #(last argument constrains the training to patterns only occuring in that model, i.e the intersectino of these models, saves heaps of space)

outfiles = []
for i in range(1,6):
    outfiles.append( open(outdir + str(i) + "_grams.txt",'w',encoding='utf-8') )

for ngram, count in patternmodel.items():
    i = len(ngram)
    anchorcount = anchormodel[ngram]
    outfiles[i-1].write( ngram.tostring(classdecoder) + "\t"+ str(anchorcount)  + "\t" + str(count) + "\t" + str(anchorcount / count)+ "\n" )

for outfile in outfiles:
    outfile.close()
