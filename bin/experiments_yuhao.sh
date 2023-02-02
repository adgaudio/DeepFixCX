#!/usr/bin/env bash
# a script to reproduce our experiments.

set -e
set -u
# set -o pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"

. ./bin/activate
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

V=111  # experiment version number


# below, some examples
C3() {
  # chexpert baselines
  cat <<EOF
  ${V}.C3.chexpert_small.resnet18.baseline.fromscratch    python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:untrained:1:14
  ${V}.C3.chexpert_small.resnet18.baseline.imagenet    python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:imagenet:1:14
EOF
  # ${V}.C3.chexpert.resnet18.baseline.fromscratch    python deepfixcx/train.py --deepfixcx off --dset chexpert:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:untrained:1:14
  # ${V}.C3.chexpert.resnet18.baseline.imagenet    python deepfixcx/train.py --deepfixcx off --dset chexpert:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:imagenet:1:14
}

# model spec is:   mlp_channels : in_ch : out_ch : wavelet_levels : patch_size : mlp_depth

C4() {
  # chexpert: experiment with varying mlp middle channel width.
  local default_cmd=" python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.2:.1 --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none"
  for ch_mid in 300 500 700 ; do
    echo ${V}.C4.chexpert_small.waveletmlp:$ch_mid:none $default_cmd --model waveletmlp:${ch_mid}:1:14:9:3:2
  done
}
C5_vecattn_regularizer() {
  # chexpert: experiment with regularizer on or off.
  local default_cmd=" python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.8:.2 --opt Adam:lr=0.003 --model waveletmlp:500:1:14:9:3:2 --lossfn chexpert_uignore "
  # experiment id has format:  mid-channels:patch_size:lambda
  cat <<EOF
  ${V}.C5.chexpert_small.waveletmlp:500:3:2:0 $default_cmd --loss_reg none
  ${V}.C5.chexpert_small.waveletmlp:500:3:2:1 $default_cmd --loss_reg deepfixcxmlp:1
  ${V}.C5.chexpert_small.waveletmlp:500:3:2:.5 $default_cmd --loss_reg deepfixcxmlp:.5
EOF
}

C6_vary_patch_size() {
  for p in 1 3 32 64; do
echo "${V}.C6.10%dset.patch$p" python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.1:.01 --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model waveletmlp:700:1:14:9:$p:2
  done
}
C6_vary_patch_size_part2() {
  for p in 128 ; do
echo "${V}.C6.10%dset.patch$p" python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.1:.01 --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model waveletmlp:300:1:14:9:$p:2
  done
}

C7() {
  # new tests with wavelet packet
# (7,1,1), (7,1,3), (6,3,1), (2,32,1)
for model in "waveletmlp:700:1:14:7:1:1:3" "waveletmlp:700:1:14:7:1:3:3" "waveletmlp:700:1:14:6:3:1:3"  "waveletmlp:700:1:14:2:30:1:3" ; do
  echo "${V}.C7.$model" python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.1:.01 --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model $model
done
}

C8() {
  # baseline, 1% of training data
  cat <<EOF
  ${V}.C8.resnet18.baseline.fromscratch    python deepfixcx/train.py --deepfixcx off --dset chexpert_small:.1:.01 --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:untrained:1:14
EOF
}

Y1() {
  # Show that our compression model hides identifying information.
  N=5000  # Use 5k identities

  # resnet18 baseline
  echo "${V}.Y1.resnet18.baseline.fromscratch python deepfixcx/train.py --dset chexpert_small_ID:N:.01:.01 --opt Adam:lr=0.001 --lossfn chexpert_identity:N --loss_reg none --model resnet18:untrained:1:N"
  # our model with  patch_size=1 --loss_reg none
  # ...todo:   echo ${V}.Y1.SOMETHiNG  python deepfixcx/train.py ...FILL IN HERE.
  # our model with  patch_size=30 --loss_reg none
  # ...todo
  # our model with  patch_size=30 --loss_reg deepfixcxmlp:1
  # ...todo

}

# C3 | run_gpus 3
# ( C5; C4 ) | run_gpus 4
# C6_vary_patch_size | run_gpus 3
# C6_vary_patch_size_part2 | run_gpus 1
# done
# C7 | run_gpus 4
# C8 | run_gpus 1
Y1 | run_gpus 1  # if this fails, "conda install redis" and then start redis-server.
