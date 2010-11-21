/*
   Copyright 2010 Joseph Reisinger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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

RESULTS_PATH=RESULTS_NCRP
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
python $CONDORIZER $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_gamma=1.0 --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --ncrp_depth=$DEPTH --sample_lag=100 --random_seed=$SEED --convergence_interval=2000 out
