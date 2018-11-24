ELMO_PATH=/home/tcastrof/.local/lib/python3.6/site-packages/allennlp

python3 elmo.py

echo "processing training set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/elmo/train/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/train/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/elmo/dev/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/dev/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016 without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/elmo/test2016/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/test2016/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017 without stopwords and punctuation marks..."
INPUT_PATH=/roaming/tcastrof/elmo/test2017/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/test2017/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing training set..."
INPUT_PATH=/roaming/tcastrof/elmo/train_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/train_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set..."
INPUT_PATH=/roaming/tcastrof/elmo/dev_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/dev_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2016..."
INPUT_PATH=/roaming/tcastrof/elmo/test2016_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/test2016_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing test set 2017..."
INPUT_PATH=/roaming/tcastrof/elmo/test2017_full/sentences.txt
OUTPUT_PATH=/roaming/tcastrof/elmo/test2017_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top