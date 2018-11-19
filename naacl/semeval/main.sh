# Main script for results of NAACL 2019 paper

PROJECT_PATH=/home/tcastrof/Questions/DiscoSumo/naacl
SEMEVAL_PATH=/home/tcastrof/Questions/DiscoSumo/naacl/semeval

# SEMEVAL
echo "preprocessing corpus..."
cd $SEMEVAL_PATH
python3 preprocessing.py

echo "train alignments with GIZA"
cd $SEMEVAL_PATH/alignments
sh run.sh

echo "train wordvec vectors"
cd $SEMEVAL_PATH/word2vec
python3 word2vec.py

echo "train elmo vectors"
cd $SEMEVAL_PATH/elmo
sh run.sh

cd $SEMEVAL_PATH
python3 evaluate.py