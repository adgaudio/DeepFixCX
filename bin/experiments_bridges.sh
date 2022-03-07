#!/usr/bin/env bash
# a script to reproduce our experiments.

echo For running on bridges
date

# load the experiments.sh file (but ignore the last two lines so nothing runs)
source <(cat ./bin/experiments.sh|head -n-1 |tail -n+2)

# copy the dataset to $LOCAL (an ssd) or $RAMDISK (54 gigs of ram)
# rsync -a --progress data/CheXpert-v1.0-small $RAMDISK
# ln -sf $RAMDISK/CheXpert-v1.0-small ./data/

export num_workers=0
export batch_size=400
# run a job
# C19 | parallel "echo  run {}" | parallel -j 1
compute_normalization | grep chexpert_small | parallel -j 40 "{} --device cpu"
# export batch_size=40
# compute_normalization | grep -v chexpert_small | parallel -j 40 "{} --device cpu"

echo done
date
exit

