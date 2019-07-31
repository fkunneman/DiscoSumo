ELMO_PATH=/home/tcastrof/.local/lib/python3.6/site-packages/allennlp

python3 elmo.py

#############################################################################
echo "processing training set lowercased without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set lowercased without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 lowercased without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 lowercased without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top
#############################################################################
echo "processing training set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.stop.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.stop.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top
#############################################################################
echo "processing training set lowercased without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set lowercased without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 lowercased without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 lowercased without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top
#############################################################################
echo "processing training set lowercased without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set lowercased without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 lowercased without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 lowercased without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top
#############################################################################
echo "processing training set without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 without punctuation marks..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.punct/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.punct/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top
#############################################################################
echo "processing training set without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 without stopwords..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.stop/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.stop/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top
#############################################################################
echo "processing training set lowercased..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/train.lower/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set lowercased..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/dev.lower/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 lowercased..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2016.lower/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 lowercased..."
INPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/semeval/elmo/test2017.lower/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top