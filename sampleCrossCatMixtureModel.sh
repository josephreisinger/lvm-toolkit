#!/bin/bash 

DATAFILE=$1
M=$2
ALPHA=$3
ETA=$4
XI=$5
CONVERGENCE=$6
SEED=$7
NOISE=${8}
KMAX=${9}

MEM=4G

echo $KMAX MAX_CLUSTERS
echo $M VIEWS
echo $SEED SEED
echo $NOISE NOISE

if [ $MEM = "4G" ]; then
    CONDORIZER=/projects/nn/joeraii/condorizer.py
    BINARY=/scratch/cluster/joeraii/ncrp/sampleCrossCatMixtureModel
    echo Using 4G machines
elif [ $MEM = "8G" ]; then
    CONDORIZER=/projects/nn/joeraii/condorizer-8G.py
    BINARY=/scratch/cluster/joeraii/ncrp/sampleCrossCatMixtureModel64
    echo Using 8G machines
else
    echo Error parsing memory requirement
    exit
fi

SHORN_DATAFILE=${DATAFILE##*/}

BASE_RUN_PATH=XCAT-$SHORN_DATAFILE-${KMAX}KMAX-${M}M-$ALPHA-$ETA-$XI-noise_is_${NOISE}
FULL_RUN_PATH=$BASE_RUN_PATH/$SEED

ORIGINAL_PATH=`pwd`

mkdir $BASE_RUN_PATH
cd $BASE_RUN_PATH
mkdir $SEED
cd $SEED


python $CONDORIZER $BINARY \
 --mm_datafile=$ORIGINAL_PATH/$DATAFILE --KMAX=$KMAX --M=$M --mm_alpha=$ALPHA --eta=$ETA \
 --max_gibbs_iterations=2000 --cc_xi=$XI --random_seed=$SEED \
 --sample_lag=0 \
 --cc_include_noise_view=$NOISE \
 --cc_feature_move_rate=1.0 \
 --convergence_interval=$CONVERGENCE \
 out
