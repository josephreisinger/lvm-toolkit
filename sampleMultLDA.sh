#!/bin/bash

DATAFILE=$1
TOPICS=$2
CRP_ALPHA=$3
CRP_ETA=$4
SEED=$5

CONDORIZER=/projects/nn/joeraii/condorizer.py
BINARY=/scratch/cluster/joeraii/ncrp/sampleMultNCRP

SHORN_DATAFILE=${DATAFILE##*/}

RESULTS_PATH=RESULTS_LDA
BASE_RUN_PATH=RUN-$SHORN_DATAFILE-LDA$TOPICS-$CRP_ALPHA-$CRP_ETA
FULL_RUN_PATH=$RESULTS_PATH/$BASE_RUN_PATH/$SEED

ORIGINAL_PATH=`pwd`

mkdir $RESULTS_PATH
cd $RESULTS_PATH
mkdir $BASE_RUN_PATH
cd $BASE_RUN_PATH
mkdir $SEED
cd $SEED

#GLOG_logtostderr=1 $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --ncrp_max_branches=1 --ncrp_depth=$TOPICS --sample_lag=100 --random_seed=$SEED out
python $CONDORIZER $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --ncrp_max_branches=1 --ncrp_depth=$TOPICS --sample_lag=100  --convergence_interval=2000 --random_seed=$SEED out
