# moses path
MOSESDIR=/home/tcastrof/workspace/mosesdecoder

# MGIZA path
MGIZA=/home/tcastrof/workspace/mgiza/mgizapp/bin

#  path
ALIGNMENTS_PATH=/roaming/tcastrof/semeval/alignments

python3 alignments.py

#############################################################################
# lower / stop / punct
cd $ALIGNMENTS_PATH/align.lower.stop.punct
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

#############################################################################
# lower / stop
cd $ALIGNMENTS_PATH/align.lower.stop
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

#############################################################################
# lower / punct
cd $ALIGNMENTS_PATH/align.lower.punct
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

#############################################################################
# stop / punct
cd $ALIGNMENTS_PATH/align.stop.punct
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

#############################################################################
# lower
cd $ALIGNMENTS_PATH/align.lower
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
#############################################################################
# stop
cd $ALIGNMENTS_PATH/align.stop
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
#############################################################################
# punct
cd $ALIGNMENTS_PATH/align.punct
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
