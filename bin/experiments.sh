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
lockfile_maxsuccesses=5
lockfile_maxconcurrent=5
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


I1 | expand 3 | run_gpus 5
I2 | run_gpus 5
