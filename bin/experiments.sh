#!/usr/bin/env bash
# a script to reproduce our experiments.

set -e
set -u
# set -o pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"

. ./bin/bash_lib.sh

expand() {
  N=${1}
  while read -r data ; do
    for i in $(seq 1 "$N")  ; do
      echo "$data"
    done
  done
}

# lockfile_ignore=true  # disable lockfile
lockfile_maxsuccesses=6
lockfile_maxconcurrent=6
lockfile_maxfailures=1

V=1  # experiment version number


I1() {
local deepfix="reinit:2:.2:1"  # N:P:R
local e1="${V}.I1.baseline"
local e2="${V}.I1.$deepfix"
cat <<EOF
$e1 python deepfix/train.py  --experiment_id $e1 --deepfix off
$e2 python deepfix/train.py  --experiment_id $e2 --deepfix $deepfix
EOF
}

I2() {
  # grid search over N and P
for P in .05 .1 .2 .3 .4 ; do
  for N in 1 2 3 ; do
  local deepfix="reinit:$N:$P:1"
  local e2="${V}.I2.$deepfix"
cat <<EOF
$e2 python deepfix/train.py  --experiment_id $e2 --deepfix $deepfix
EOF
done ; done
}

I3() {
  # find the distribution of weights for each scalar parameter in the network
  for m in nth_most_salient nth_weight ; do
  for sm in "weight*grad" "grad" "weight" ; do
cat <<EOF
python bin/find_distribution.py --mode '$m' --saliency_mode '$sm' --base_dir "results/${V}.I3"
EOF
  done
  done
}
I3_part2() {
  # initialize model from a histogram for each scalar parameter (deepfix dhist method)
  # local fpout=./results/${V}.I3
  # mkdir -p "$fpout"
  # find ./histograms -name "hist_*.pth" | parallel echo \
    # python deepfix/train.py bin/eval_distribution.py "{}" "${fpout}/{/.}.csv"
  local e1="${V}.I3.baseline"
  local e2="${V}.I3.dhist"
  local e3="${V}.I3.dfhist"
  local e4="${V}.I3.fixed"
  local params=" --model resnet18:untrained:3:3 "
  cat <<EOF
$e1 python deepfix/train.py     --experiment_id $e1     --deepfix off $params
$e1.imagenet python deepfix/train.py     --experiment_id $e1.imagenet     --deepfix off
EOF
cat <<EOF
$e2.sg python deepfix/train.py  --experiment_id $e2.sg  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsgrad_resnet18:untrained:3:3.pth $params
$e2.sw python deepfix/train.py  --experiment_id $e2.sw  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsweight_resnet18:untrained:3:3.pth $params
$e2.swg python deepfix/train.py --experiment_id $e2.swg --deepfix dhist:./results/${V}.I3/histograms/hist_nth_most_salient_resnet18:untrained:3:3.pth $params
$e2.ng python deepfix/train.py  --experiment_id $e2.ng  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_weight.wsgrad_resnet18:untrained:3:3.pth $params
$e2.nw python deepfix/train.py  --experiment_id $e2.nw  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_weight.wsweight_resnet18:untrained:3:3.pth $params
$e2.nwg python deepfix/train.py --experiment_id $e2.nwg --deepfix dhist:./results/${V}.I3/histograms/hist_nth_weight_resnet18:untrained:3:3.pth $params
EOF
cat <<EOF
$e3.sg python deepfix/train.py  --experiment_id $e3.sg  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsgrad_resnet18:untrained:3:3.pth $params
$e3.sw python deepfix/train.py  --experiment_id $e3.sw  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsweight_resnet18:untrained:3:3.pth $params
$e3.swg python deepfix/train.py --experiment_id $e3.swg --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_most_salient_resnet18:untrained:3:3.pth $params
$e3.ng python deepfix/train.py  --experiment_id $e3.ng  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_weight.wsgrad_resnet18:untrained:3:3.pth $params
$e3.nw python deepfix/train.py  --experiment_id $e3.nw  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_weight.wsweight_resnet18:untrained:3:3.pth $params
$e3.nwg python deepfix/train.py --experiment_id $e3.nwg --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_weight_resnet18:untrained:3:3.pth $params
EOF
cat <<EOF
$e4 python deepfix/train.py     --experiment_id $e4     --deepfix fixed $params
$e4.imagenet python deepfix/train.py     --experiment_id $e4.imagenet --deepfix fixed
EOF
}


# I1 | expand 3 | run_gpus 5
# I2 | run_gpus 5
# I3 I3_part2| parallel -j 5
I3_part2 | run_gpus 5
