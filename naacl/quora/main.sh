# Main script for results of NAACL 2019 paper

PROJECT_PATH=/home/tcastrof/DiscoSumo/naacl
QUORA_PATH=/home/tcastrof/DiscoSumo/naacl/quora

# QUORA
echo "preprocessing corpus..."
cd $QUORA_PATH
python3 preprocessing.py

#echo "train alignments with GIZA"
#cd $QUORA_PATH/alignments
#sh run.sh

echo "train wordvec vectors"
cd $QUORA_PATH/word2vec
python3 word2vec.py

#echo "train elmo vectors"
#cd $QUORA_PATH/elmo
#sh run.sh

#cd $QUORA_PATH
#python3 evaluate.py