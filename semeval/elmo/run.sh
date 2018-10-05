ELMO_PATH=/home/tcastrof/.local/lib/python3.6/site-packages/allennlp

echo "processing training set..."
INPUT_PATH=train/sentences.txt
OUTPUT_PATH=train/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --average

echo "processing dev set..."
INPUT_PATH=dev/sentences.txt
OUTPUT_PATH=dev/elmovectors.hdf5

python3 $ELMO_PATH/run.py elmo \
                          $INPUT_PATH \
                          $OUTPUT_PATH \
                          --average