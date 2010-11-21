#!/bin/bash

DATAFILE=$1
DEPTH=$2
CRP_ALPHA=$3
CRP_ETA=$4
SEED=$5
MEM=$6

if [ $MEM = "4G" ]; then
    CONDORIZER=/projects/nn/joeraii/condorizer.py
    BINARY=/scratch/cluster/joeraii/ncrp/sampleMultNCRP
    echo Using 4G machines
elif [ $MEM = "8G" ]; then
    CONDORIZER=/projects/nn/joeraii/condorizer-8G.py
    BINARY=/scratch/cluster/joeraii/ncrp/sampleMultNCRP64
    echo Using 8G machines
else
    echo Error parsing memory requirement
    exit
fi

SHORN_DATAFILE=${DATAFILE##*/}

RESULTS_PATH=RESULTS_PRIX_FIXE
BASE_RUN_PATH=RUN-$SHORN_DATAFILE-DEPTH$DEPTH-$CRP_ALPHA-$CRP_ETA
FULL_RUN_PATH=$RESULTS_PATH/$BASE_RUN_PATH/$SEED

ORIGINAL_PATH=`pwd`

mkdir $RESULTS_PATH
cd $RESULTS_PATH
mkdir $BASE_RUN_PATH
cd $BASE_RUN_PATH
mkdir $SEED
cd $SEED

#GLOG_logtostderr=1 GLOG_v=2 $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --ncrp_depth=$DEPTH --sample_lag=100 --random_seed=$SEED out
python $CONDORIZER $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_prix_fixe=true --ncrp_gamma=1.0 --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --ncrp_depth=$DEPTH --sample_lag=100 --random_seed=$SEED --convergence_interval=2000 out
