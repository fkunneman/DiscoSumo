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

**Results in the new dataset with no categorization (Word2Vec - Dimention: 300 - Window: 10):**

Fold  1 = upper:  0.32 / BM25:  0.24 / Translation:  0.26 / Softcosine:  0.22 / Ensemble:  0.24

Fold  2 = upper:  0.48 / BM25:  0.30 / Translation:  0.33 / Softcosine:  0.26 / Ensemble:  0.29

Fold  3 = upper:  0.41 / BM25:  0.29 / Translation:  0.33 / Softcosine:  0.27 / Ensemble:  0.28

Fold  4 = upper:  0.47 / BM25:  0.31 / Translation:  0.36 / Softcosine:  0.34 / Ensemble:  0.36

Fold  5 = upper:  0.39 / BM25:  0.23 / Translation:  0.24 / Softcosine:  0.20 / Ensemble:  0.26

Averaging = upper:  - / BM25:  0.27 / **Translation:  0.30** / Softcosine:  0.26 / Ensemble:  0.29

**Results in the new dataset with categorization (Word2Vec - Dimention: 300 - Window: 10):**

Fold  1 = upper:  0.36 / BM25:  0.25 / Translation:  0.29 / Softcosine:  0.24 / Ensemble:  0.29

Fold  2 = upper:  0.33 / BM25:  0.23 / Translation:  0.22 / Softcosine:  0.20 / Ensemble:  0.20

Fold  3 = upper:  0.45 / BM25:  0.26 / Translation:  0.33 / Softcosine:  0.27 / Ensemble:  0.32

Fold  4 = upper:  0.49 / BM25:  0.30 / Translation:  0.33 / Softcosine:  0.30 / Ensemble:  0.31

Fold  5 = upper:  0.44 / BM25:  0.32 / Translation:  0.33 / Softcosine:  0.29 / Ensemble:  0.33

Averaging = upper:  - / BM25:  0.27 / **Translation:  0.30** / Softcosine:  0.26 / Ensemble:  0.29