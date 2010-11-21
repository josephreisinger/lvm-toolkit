#!/bin/bash

CONDORIZER=/projects/nn/joeraii/condorizer-8G.py
BINARY=/scratch/cluster/joeraii/ncrp/samplePrecomputedFixedNCRP64

DATAFILE=$1
TOPIC_ASSIGNMENTS_FILE=$2
NOISE_TOPICS=$3
CRP_ALPHA=$4
CRP_ETA=$5
SEED=$6

SHORN_DATAFILE=${DATAFILE##*/}
SHORN_TOPIC_ASSIGNMENTS_FILE=${TOPIC_ASSIGNMENTS_FILE##*/}

RESULTS_PATH=RESULTS_FIXED
BASE_RUN_PATH=RUN-$SHORN_DATAFILE-$SHORN_TOPIC_ASSIGNMENTS_FILE-$NOISE_TOPICS-$CRP_ALPHA-$CRP_ETA
FULL_RUN_PATH=$RESULTS_PATH/$BASE_RUN_PATH/$SEED

ORIGINAL_PATH=`pwd`

mkdir $RESULTS_PATH
cd $RESULTS_PATH
mkdir $BASE_RUN_PATH
cd $BASE_RUN_PATH
mkdir $SEED
cd $SEED

# echo $BINARY --crp_datafile=$ORIGINAL_PATH/$DATAFILE --crp_alpha=$CRP_ALPHA --crp_eta=$CRP_ETA --topic_assignments_file=$ORIGINAL_PATH/$TOPIC_ASSIGNMENTS_FILE --additional_noise_topics=$NOISE_TOPICS  --use_dag=true --sample_lag=50 --random_seed=$SEED out
python $CONDORIZER $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --topic_assignments_file=$ORIGINAL_PATH/$TOPIC_ASSIGNMENTS_FILE --additional_noise_topics=$NOISE_TOPICS  --use_dag=true --sample_lag=100 --random_seed=$SEED --cull_unique_topics=false out
# GLOG_logtostderr=1 $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --topic_assignments_file=$ORIGINAL_PATH/$TOPIC_ASSIGNMENTS_FILE --additional_noise_topics=$NOISE_TOPICS  --use_dag=true --sample_lag=100 --random_seed=$SEED out
