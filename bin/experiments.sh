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

I3_part1() {
  # find the distribution of weights for each scalar parameter in the network
  for m in nth_most_salient nth_weight ; do
  for sm in "weight*grad" "grad" "weight" ; do
cat <<EOF
python bin/find_distribution.py --mode '$m' --saliency_mode '$sm' --base_dir "results/${V}.I3"
EOF
  done
  done
  # also show the distribution for the quantiles of a pre-trained network
cat<<EOF
python bin/find_distribution_single_model.py --model 'resnet18:untrained:3:3' --base_dir "results/${V}.I3/histograms_single_model"
python bin/find_distribution_single_model.py --model 'resnet18:imagenet:3:3' --base_dir "results/${V}.I3/histograms_single_model"
EOF
}
I3_part2() {
  # initialize model from a histogram for each scalar parameter obtained by analyzing untrained models (deepfix dhist method) 
  # local fpout=./results/${V}.I3
  # mkdir -p "$fpout"
  # find ./histograms -name "hist_*.pth" | parallel echo \
    # python deepfix/train.py bin/eval_distribution.py "{}" "${fpout}/{/.}.csv"
  local e1="${V}.I3.baseline"
  local e2="${V}.I3.dhist"
  local e3="${V}.I3.dfhist"
  local e4="${V}.I3.fixed"
  local e5="${V}.I3.dshist"
  local params=" --model resnet18:untrained:3:3 "
  cat <<EOF
run $e1 python deepfix/train.py     --experiment_id $e1     --deepfix off $params
run $e1.imagenet python deepfix/train.py     --experiment_id $e1.imagenet     --deepfix off
EOF
cat <<EOF
run $e2.sg python deepfix/train.py  --experiment_id $e2.sg  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsgrad_resnet18:untrained:3:3.pth $params
run $e2.sw python deepfix/train.py  --experiment_id $e2.sw  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsweight_resnet18:untrained:3:3.pth $params
run $e2.swg python deepfix/train.py --experiment_id $e2.swg --deepfix dhist:./results/${V}.I3/histograms/hist_nth_most_salient_resnet18:untrained:3:3.pth $params
run $e2.ng python deepfix/train.py  --experiment_id $e2.ng  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_weight.wsgrad_resnet18:untrained:3:3.pth $params
run $e2.nw python deepfix/train.py  --experiment_id $e2.nw  --deepfix dhist:./results/${V}.I3/histograms/hist_nth_weight.wsweight_resnet18:untrained:3:3.pth $params
run $e2.nwg python deepfix/train.py --experiment_id $e2.nwg --deepfix dhist:./results/${V}.I3/histograms/hist_nth_weight_resnet18:untrained:3:3.pth $params
EOF
cat <<EOF
run $e3.sg python deepfix/train.py  --experiment_id $e3.sg  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsgrad_resnet18:untrained:3:3.pth $params
run $e3.sw python deepfix/train.py  --experiment_id $e3.sw  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_most_salient.wsweight_resnet18:untrained:3:3.pth $params
run $e3.swg python deepfix/train.py --experiment_id $e3.swg --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_most_salient_resnet18:untrained:3:3.pth $params
run $e3.ng python deepfix/train.py  --experiment_id $e3.ng  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_weight.wsgrad_resnet18:untrained:3:3.pth $params
run $e3.nw python deepfix/train.py  --experiment_id $e3.nw  --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_weight.wsweight_resnet18:untrained:3:3.pth $params
run $e3.nwg python deepfix/train.py --experiment_id $e3.nwg --deepfix dfhist:./results/${V}.I3/histograms/hist_nth_weight_resnet18:untrained:3:3.pth $params
EOF
cat <<EOF
run $e4 python deepfix/train.py     --experiment_id $e4     --deepfix fixed $params
run $e4.imagenet python deepfix/train.py     --experiment_id $e4.imagenet --deepfix fixed
EOF
# what happens when you initialize to the distribution of imagenet?  if you lose the pre-training benefit, then the value is in the path through layers.
# ... the model doesn't learn basically at all.
cat <<EOF
run $e5 python deepfix/train.py --experiment_id $e5 --deepfix dfhist:./results/${V}.I3/histograms_single_model/resnet18:imagenet:3:3_weightgrad.pth $params
EOF
}

I4() {
  # try initializing to a U-shaped distribution (Beta distribution with alpha and beta < 1)
  for pretraining in untrained imagenet ; do
    for beta in .01 .2 .5 .8 ; do
    local e="${V}.I4.beta.${beta}.${pretraining}"
    echo run $e python deepfix/train.py --experiment_id $e --model resnet18:$pretraining:3:3 --deepfix beta:$beta:$beta
  done
done
}


E1() {
  echo run E1.resnet18.IntelMobileODT  python deepfix/train.py  --experiment_id E1.resnet18.IntelMobileODT  --deepfix off
  # for layer in pointwise spatial ; do
    # for model in resnet18:imagenet:1 efficientnet-b0:imagenetadv:1 ; do
      # for dset in CheXpert ; do
        # echo python bin/vis_p2.py --layer $layer --model $model --dset $dset --allplots
    # done ; done
  # done
  echo python bin/vis_p2.py --layer pointwise --model resnet18:IntelMobileODT:3 --dset IntelMobileODT --allplots
}

E2() {
  local args=" --dset intel_mobileodt:train+additional:val:test:v1 --model resnet18:imagenet:3:3 --epochs 300"
  cat <<EOF
  E2.IntelMobileODT.resnet18.baseline          python deepfix/train.py --deepfix off                $args
  E2.IntelMobileODT.resnet18.ghaarconv2d:      python deepfix/train.py --deepfix ghaarconv2d:       $args
  E2.IntelMobileODT.resnet18.ghaarconv2d:conv1 python deepfix/train.py --deepfix ghaarconv2d:conv1  $args
EOF
}


# I1 | expand 3 | run_gpus echo 5
# I2 | run_gpus 5
# I3_part1 | parallel -j 5
# I3_part2 I3_part2 I3_part2 | parallel -j 5
# I3_part2 | run_gpus 5
# I4 | parallel -j 1

# E1 | parallel -j 1
# for i in {1..6} ; do
E2 | run_gpus 3
# done
