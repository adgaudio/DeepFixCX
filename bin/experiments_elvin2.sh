#!/usr/bin/env bash
# a script to reproduce our experiments.

echo For running on bridges
date

. ./bin/activate
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
lockfile_maxsuccesses=1
lockfile_maxconcurrent=1
lockfile_maxfailures=1

E=10 # experiment version number
export num_workers=0

# load the experiments.sh file (but ignore the last two lines so nothing runs)
E1() {
  # experiment over varying wavelet types
  python <<EOF
for wavelet_level in 1,2,3,4,5,6,7,8,9:
    for patch_size in 1,3,5,9,19,37,79,115,160:
#         waveletmlpV2:1:14:coif1:5:5:l1
        model = f'waveletmlpV2:1:14:coif1:{wavelet_level}:{patch_size}:l1'
        print(f""" ${E}.E1.{model}     python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model}""")
EOF
}

# run a job
# E10 | run_gpus 4
E1 | parallel "echo  run {}" | parallel -j 5

echo done
date
exit