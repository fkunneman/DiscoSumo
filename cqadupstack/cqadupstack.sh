# moses path
MOSESDIR=/home/tcastrof/workspace/mosesdecoder

# MGIZA path
MGIZA=/home/tcastrof/workspace/mgiza

# cqadupstack path
CQADUBSTACK=/home/tcastrof/Question/cqadupstack
CATEGORY=android
TRANSLATION=/home/tcastrof/Question/cqadupstack/$CATEGORY/translation

cd $CQADUBSTACK
python3 cqadupstack_align.py

cd $TRANSLATION
cat $TRANSLATION/$CATEGORY.de | \
sed 's/[A-Z]/\L&/g' | \
perl $MOSESDIR/scripts/tokenizer/escape-special-chars.perl | \
perl $MOSESDIR/scripts/tokenizer/normalize-punctuation.perl > train.de

cat $TRANSLATION/$CATEGORY.en | \
sed 's/[A-Z]/\L&/g' | \
perl $MOSESDIR/scripts/tokenizer/escape-special-chars.perl | \
perl $MOSESDIR/scripts/tokenizer/normalize-punctuation.perl > train.en

perl $MOSESDIR/scripts/training/train-model.perl \
    -root-dir . \
    --corpus train \
    -mgiza -mgiza-cpus 4 \
    --max-phrase-length 4 \
    -external-bin-dir $MGIZA \
    --f de --e en \
    --parallel \
    --last-step 8 \
    --distortion-limit 6

cd $CQADUBSTACK
python3 cqadupstack.py