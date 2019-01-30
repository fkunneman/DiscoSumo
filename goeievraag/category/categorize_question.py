import sys

from goeievraag.category.qcat import QCat

model_file = sys.argv[1]
label_encoder_file = sys.argv[2]
category2id_file = sys.argv[3]
vocabulary_file = sys.argv[4]

qc = QCat(model_file,label_encoder_file,category2id_file,vocabulary_file)

test_questions = ["Kunnen we volgende week weer schaatsen op natuurijs",
                  "Wat is het lekkerste recept voor boerenkool",
                  "Hoeveel kleuren heeft de regenboog",
                  "Wat is de symbolische betekenis van de kip die de vrouw vasthoudt op het schilderij De Nachtwacht",
                  "waar kan ik in amsterdam het best een dwerg hamster aanschaffen",
                  "Waarom zie je nooit babyduifjes",
                  "Hoe krijg je een weggelopen konijn ( ontsnapt ) weer terug",
                  "Wat is het synoniem voor synoniem",
                  "wat s de reden dat vogels niet vastvriezen aan een ijsschots",
                  "Als een winkel 24 uur per dag en 365 dagen per jaar geopend is , waarom zit er dan een slot op de deur"]

print('Now categorizing questions')
results = qc.main(test_questions,5)
for i,result in enumerate(results):
    print('TOP 5 categories for question',test_questions[i],':',result)
