# moses path
MOSESDIR=/home/tcastrof/workspace/mosesdecoder

# MGIZA path
MGIZA=/home/tcastrof/workspace/mgiza

#  path
SEMEVAL=/home/tcastrof/Question/DiscoSumo/naacl/semeval/alignments

python3 alignments.py

# clean corpus
perl $MOSESDIR/scripts/training/clean-corpus-n.perl semeval de en train 1 80

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