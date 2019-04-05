# GoeieVraag

## Initialization

The following command will process all the files needed to execute the API.
```
python3 init.py
```

## Question Similarity

The main file for GoeieVraag, with the question similarity metrics is `main.py`

## API

The api was coded in [Flask](http://flask.pocoo.org/).

## Corpus Info

1. Old dataset (`question_parsed.json`): 627.340 questions
2. New dataset (`question_parsed_new.json`): 423,149 questions
3. Union (`question_parsed_final.json`): 639,085 questions
4. Seed questions: 117,076 questions. These questions are the ones extracted from the union dataset which has at least one star and one answer
 
## Evaluation

**Results in the old dataset with no categorization:**

Fold  1 = upper:  0.49 / BM25:  0.36 / Translation:  0.39 / Softcosine:  0.32 / Ensemble:  0.34

Fold  2 = upper:  0.24 / BM25:  0.11 / Translation:  0.17 / Softcosine:  0.16 / Ensemble:  0.16

Fold  3 = upper:  0.41 / BM25:  0.3 / Translation:  0.27 / Softcosine:  0.23 / Ensemble:  0.25

Fold  4 = upper:  0.38 / BM25:  0.2 / Translation:  0.26 / Softcosine:  0.2 / Ensemble:  0.22

Fold  5 = upper:  0.55 / BM25:  0.39 / Translation:  0.42 / Softcosine:  0.43 / Ensemble:  0.43

Averaging = upper:  0.41 / BM25:  0.27 / **Translation:  0.30** / Softcosine:  0.27 / Ensemble:  0.28