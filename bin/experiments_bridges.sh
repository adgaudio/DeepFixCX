#!/usr/bin/env bash
# a script to reproduce our experiments.

echo For running on bridges
date

# load the experiments.sh file (but ignore the last two lines so nothing runs)
source <(cat ./bin/experiments.sh|head -n-1 |tail -n+2)

export num_workers=0
export batch_size=40
# run a job
C16 | parallel "echo  run {}" | parallel -j 10

echo done
date
exit
