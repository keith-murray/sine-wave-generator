#!/bin/bash

source /etc/profile
module load anaconda/2023a-pytorch

echo "My task ID: $LLSUB_RANK"
echo "Number of Tasks: $LLSUB_SIZE"

python /home/gridsan/kmurray/sinusoidal-network/src/pipeline.py $LLSUB_RANK /home/gridsan/kmurray/sinusoidal-network/results/experiment_1
