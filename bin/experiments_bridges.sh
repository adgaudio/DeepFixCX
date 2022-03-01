#!/usr/bin/env bash
# a script to reproduce our experiments.

echo For running on bridges
date

# load the experiments.sh file (but ignore the last two lines so nothing runs)
source <(cat ./bin/experiments.sh|head -n-1 |tail -n+2)

export num_workers=6
export batch_size=2000
# run a job
C19 | parallel "echo  run {}" | parallel -j 1

echo done
date
exit

