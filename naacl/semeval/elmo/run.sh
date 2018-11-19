ELMO_PATH=/home/tcastrof/.local/lib/python3.6/site-packages/allennlp

python3 elmo.py

echo "processing training set without stopwords and punctuation marks..."
INPUT_PATH=train/sentences.txt
OUTPUT_PATH=train/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set without stopwords and punctuation marks..."
INPUT_PATH=dev/sentences.txt
OUTPUT_PATH=dev/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing training set..."
INPUT_PATH=train_full/sentences.txt
OUTPUT_PATH=train_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top

echo "processing dev set..."
INPUT_PATH=dev_full/sentences.txt
OUTPUT_PATH=dev_full/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --top