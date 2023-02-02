#!/usr/bin/env bash

# fetch results from bridges, at location ~/store/deepfixcx/results/
# doesn't download checkpoints 

# usage:  ./bin/fetch_bridges "2.C12*"  # gets data from all experiments matching ./results/2.C12*

set -e
set -u

experiment_id=$1  # or a glob expression
rsync -ave ssh --exclude "*.pth" bridgesdata:store/deepfixcx/results/$experiment_id ./results_bridges/
rsync -ave ssh bridgesdata:store/deepfixcx/results/$experiment_id ./results_bridges/
