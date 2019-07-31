ELMO_PATH=/home/tcastrof/.local/lib/python3.6/site-packages/allennlp

python3 elmo.py

echo "processing training set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/quora/elmo/train/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/quora/elmo/train/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/quora/elmo/dev/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/quora/elmo/dev/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/quora/elmo/test/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/quora/elmo/test/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing training set..."
INPUT_PATH=/roaming/tcastrof/quora/elmo/train_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/quora/elmo/train_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set..."
INPUT_PATH=/roaming/tcastrof/quora/elmo/dev_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/quora/elmo/dev_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set..."
INPUT_PATH=/roaming/tcastrof/quora/elmo/test_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/quora/elmo/test_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top