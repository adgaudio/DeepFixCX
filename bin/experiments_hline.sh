#!/usr/bin/env bash
# a script to reproduce our experiments.

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

V=5  # experiment version number


HL1() {
  # cardiomegaly
# V=1: adam
# V=2: sgd, momentum=.9
# V=3: std, momentum=.9, nesterov=1
  opt="SGD:lr=0.001:momentum=.9:nesterov=1"
# cat <<EOF
# $V.HL1.rline    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model rline --opt $opt --lossfn chexpert_uignore --epochs 100
# $V.HL1.rline2_200    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model rline2_200 --opt $opt --lossfn chexpert_uignore --epochs 100
# $V.HL1.hline    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model hline --opt $opt --lossfn chexpert_uignore --epochs 100
# $V.HL1.densenet env batch_size=15 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model densenet121:untrained:1:1 --opt $opt --lossfn chexpert_uignore  --epochs 100
# EOF

# for mdl in rline2_200 rline_200heart rline_200heart2 rline1 rline2 rline3 rline3f ; do
for mdl in rline_200heart2 rhline ; do
  echo $V.HL1.$mdl.1    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model $mdl --opt $opt --lossfn chexpert_uignore --epochs 100
  echo $V.HL1.$mdl.2    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model $mdl --opt $opt --lossfn chexpert_uignore --epochs 100
  echo $V.HL1.$mdl.3    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Cardiomegaly --model $mdl --opt $opt --lossfn chexpert_uignore --epochs 100
done
}

HL2() {
  # Enlarged Cardiomediastinum
cat <<EOF
$V.HL2.rline    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Enlarged_Cardiomediastinum --model rline --opt SGD:lr=0.001:momentum=.9 --lossfn chexpert_uignore --epochs 100
$V.HL2.hline    env batch_size=1000 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Enlarged_Cardiomediastinum --model hline --opt SGD:lr=0.001:momentum=.9 --lossfn chexpert_uignore --epochs 100
$V.HL2.densenet env batch_size=15 num_workers=6 python deepfix/train.py --dset chexpert_small15k:.9:.1:Enlarged_Cardiomediastinum --model densenet121:untrained:1:1 --opt SGD:lr=0.001:momentum=.9 --lossfn chexpert_uignore  --epochs 100
EOF
# V=1: adam
# V=2: sgd, momentum=.9
}

HL3() {
  # reproduce elvin, using rhline
  python <<EOF
for n,opt in enumerate(["SGD:lr=0.0005:momentum=.7:nesterov=1:weight_decay=1e-6", ]):
  for mdl in ['hline_10', 'rhline', 'rline', 'heart', 'qrhline', 'qrhline_fast']:
    print(f''' $V.HL3.opt{n}.{mdl}.    env batch_size=2048 num_workers=6 python deepfix/train.py --dset chexpert_small:.9:.1:Cardiomegaly --model {mdl} --opt {opt} --lossfn chexpert_uignore --epochs 50 ''')
  # some tests usig the sum
  for mdl in ['sum', 'heart+sum', 'rhline+heart+sum', 'rhline+sum']:
    print(f''' $V.HL3.opt{n}.{mdl}.    env batch_size=2048 num_workers=6 python deepfix/train.py --dset chexpert_small:.9:.1:Cardiomegaly --model {mdl} --opt {opt} --lossfn chexpert_uignore --epochs 50 ''')
for n,opt in enumerate(["SGD:lr=0.0005:momentum=.7:nesterov=1:weight_decay=1e-6", "Adam:lr=0.001"]):
  for mdl in ['median+rhline+heart', ]:
    print(f''' $V.HL3.opt{n}.{mdl}.    env batch_size=340 num_workers=6 python deepfix/train.py --dset chexpert_small:.9:.1:Cardiomegaly --model {mdl} --opt {opt} --lossfn chexpert_uignore --epochs 50 ''')
EOF
echo $V.HL3.median    env batch_size=480 num_workers=6 python deepfix/train.py --dset chexpert_small:.9:.1:Cardiomegaly --model median                    --opt Adam:lr=0.001 --lossfn chexpert_uignore  --epochs 50
echo $V.HL3.densenet  env batch_size=30  num_workers=6 python deepfix/train.py --dset chexpert_small:.9:.1:Cardiomegaly --model densenet121:untrained:1:1 --opt Adam:lr=0.001 --lossfn chexpert_uignore  --epochs 50
}

HL4() {
  # tuner, SGD
  run $V.HL4 env num_workers=6 batch_size=800 python deepfix/train_tuner.py  --dset chexpert_small:.9:.1:Cardiomegaly --lossfn chexpert_uignore #--resume True
}
HL5() {
  # tuner, Adam
  run $V.HL5 env num_workers=6 batch_size=800 python deepfix/train_tuner.py  --dset chexpert_small:.9:.1:Cardiomegaly --lossfn chexpert_uignore --opt Adam:lr=tune
}



# HL1 | run_gpus 1
# HL2 | run_gpus 3
# HL3 | run_gpus 1
HL5
