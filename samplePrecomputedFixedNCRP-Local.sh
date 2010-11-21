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

BINARY=/scratch/cluster/joeraii/ncrp/samplePrecomputedFixedNCRP

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

echo "culling"

mkdir $RESULTS_PATH
cd $RESULTS_PATH
mkdir $BASE_RUN_PATH
cd $BASE_RUN_PATH
mkdir $SEED
cd $SEED

GLOG_logtostderr=1 $BINARY --ncrp_datafile=$ORIGINAL_PATH/$DATAFILE --ncrp_alpha=$CRP_ALPHA --eta=$CRP_ETA --topic_assignments_file=$ORIGINAL_PATH/$TOPIC_ASSIGNMENTS_FILE --additional_noise_topics=$NOISE_TOPICS  --use_dag=true --sample_lag=100 --random_seed=$SEED out
