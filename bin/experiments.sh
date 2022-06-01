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

V=2  # experiment version number
export num_workers=4


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

E3() {
  # baselines
  local args=' --opt Adam:lr=0.001 --epochs 300 --lossfn CrossEntropyLoss '
  cat <<EOF
  ${V}.E3.IntelMobileODT_ta.resnet18.baseline.fromscratch python deepfix/train.py --dset intel_mobileodt:train+additional:val:test:v1 --model resnet18:untrained:3:3  $args
  ${V}.E3.IntelMobileODT_ta.resnet18.baseline.imagenet    python deepfix/train.py --dset intel_mobileodt:train+additional:val:test:v1 --model resnet18:imagenet:3:3  $args
  ${V}.E3.IntelMobileODT_t.resnet18.baseline.fromscratch python deepfix/train.py --dset intel_mobileodt:train:val:test:v1 --model resnet18:untrained:3:3  $args
  ${V}.E3.IntelMobileODT_t.resnet18.baseline.imagenet    python deepfix/train.py --dset intel_mobileodt:train:val:test:v1 --model resnet18:imagenet:3:3  $args
EOF
}
E4() {
  # evaluate resnet18 baseline on wavelet transformed images.
  # see that imagenet pre-training gains are lost.
  local args=" --epochs 300"
  cat <<EOF
  ${V}.E4.IntelMobileODT_ta.waveletres18.fromscratch python deepfix/train.py --deepfix off --dset intel_mobileodt:train+additional:val:test:v1 --model waveletres18:untrained:3:3  $args
  ${V}.E4.IntelMobileODT_ta.waveletres18.imagenet    python deepfix/train.py --deepfix off --dset intel_mobileodt:train+additional:val:test:v1 --model waveletres18:imagenet:3:3  $args
  ${V}.E4.IntelMobileODT_t.waveletres18.fromscratch python deepfix/train.py --deepfix off --dset intel_mobileodt:train:val:test:v1 --model waveletres18:untrained:3:3  $args
  ${V}.E4.IntelMobileODT_t.waveletres18.imagenet    python deepfix/train.py --deepfix off --dset intel_mobileodt:train:val:test:v1 --model waveletres18:imagenet:3:3  $args
EOF
}
E5() {
  # waveletmlp
  # NOTE: this was done with the preprocessing_transform with I think 3 conv2d(...) layers, not just 1.  the initial results I have are 3 conv layers but code is now just 1 layer.
  cat <<EOF
  ${V}.E5.IntelMobileODT_ta.waveletmlp:500 python deepfix/train.py --deepfix off --dset intel_mobileodt:train+additional:val:test:v1 --model waveletmlp:500:3:3:true  --epochs 300 --opt SGD:lr=0.001
EOF
}


C3() {
  # chexpert baselines
  cat <<EOF
  ${V}.C3.chexpert_small.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert_small:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:untrained:1:14
  ${V}.C3.chexpert_small.resnet18.baseline.imagenet    python deepfix/train.py --deepfix off --dset chexpert_small:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:imagenet:1:14
EOF
  # ${V}.C3.chexpert.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:untrained:1:14
  # ${V}.C3.chexpert.resnet18.baseline.imagenet    python deepfix/train.py --deepfix off --dset chexpert:.8:.2 --loss chexpert_uignore --opt Adam:lr=0.003 --model resnet18:imagenet:1:14
}

# model spec is:   mlp_channels : in_ch : out_ch : wavelet_levels : patch_size : mlp_depth

C4() {
  # chexpert: experiment with varying mlp middle channel width.
  local default_cmd=" python deepfix/train.py --deepfix off --dset chexpert_small:.2:.1 --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none"
  for ch_mid in 300 500 700 ; do
    echo ${V}.C4.chexpert_small.waveletmlp:$ch_mid:none $default_cmd --model waveletmlp:${ch_mid}:1:14:9:3:2
  done
}
C5_vecattn_regularizer() {
  # chexpert: experiment with regularizer on or off.
  local default_cmd=" python deepfix/train.py --deepfix off --dset chexpert_small:.8:.2 --opt Adam:lr=0.003 --model waveletmlp:500:1:14:9:3:2 --lossfn chexpert_uignore "
  # experiment id has format:  mid-channels:patch_size:lambda
  cat <<EOF
  ${V}.C5.chexpert_small.waveletmlp:500:3:2:0 $default_cmd --loss_reg none
  ${V}.C5.chexpert_small.waveletmlp:500:3:2:1 $default_cmd --loss_reg deepfixmlp:1
  ${V}.C5.chexpert_small.waveletmlp:500:3:2:.5 $default_cmd --loss_reg deepfixmlp:.5
EOF
}

C6_vary_patch_size() {
  for p in 1 3 32 64; do
echo "${V}.C6.10%dset.patch$p" python deepfix/train.py --deepfix off --dset chexpert_small:.1:.01 --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model waveletmlp:700:1:14:9:$p:2
  done
}
C6_vary_patch_size_part2() {
  for p in 128 ; do
echo "${V}.C6.10%dset.patch$p" python deepfix/train.py --deepfix off --dset chexpert_small:.1:.01 --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model waveletmlp:300:1:14:9:$p:2
  done
}

C7() {
  # new tests with wavelet packet
# (7,1,1), (7,1,3), (6,3,1), (2,32,1)
for model in "waveletmlp:700:1:14:7:1:1:3" "waveletmlp:700:1:14:7:1:3:3" "waveletmlp:700:1:14:6:3:1:3"  "waveletmlp:700:1:14:2:30:1:3" ; do
  echo "${V}.C7.$model" python deepfix/train.py --deepfix off --dset chexpert_small:.1:.01 --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model $model
done
}

C8() {
  # baseline
  local V=3
  # ${V}.C8.diagnostic.densenet121.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:untrained:1:14 --epochs 80
  # ${V}.C8.Cardiomegaly.densenet121.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:Cardiomegaly --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:untrained:1:1 --epochs 80
  # ${V}.C8.Cardiomegaly.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:Cardiomegaly --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:untrained:1:1 --epochs 80

  cat <<EOF
  ${V}.C8.leaderboard.resnet18.baseline.fromscratch    env batch_size=55 num_workers=6    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:leaderboard --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:untrained:1:5 --epochs 80
  ${V}.C8.leaderboard.resnet18.baseline.imagenet       env batch_size=55 num_workers=6    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:leaderboard --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:imagenet:1:5 --epochs 80
  ${V}.C8.leaderboard.densenet121.baseline.fromscratch env batch_size=10 num_workers=6    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:leaderboard --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:untrained:1:5 --epochs 80
  ${V}.C8.leaderboard.densenet121.baseline.imagenet    env batch_size=10 num_workers=6    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:leaderboard --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:imagenet:1:5 --epochs 80
  ${V}.C8.diagnostic.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:untrained:1:14 --epochs 80
  ${V}.C8.diagnostic.resnet18.baseline.imagenet    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:imagenet:1:14 --epochs 80
  ${V}.C8.diagnostic.densenet121.baseline.fromscratch    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:untrained:1:14 --epochs 80
  ${V}.C8.diagnostic.densenet121.baseline.imagenet    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:imagenet:1:14 --epochs 80
EOF
}
# I8() {
#   local num_identities=100
#   local train_pct=.01
#   # identity model baselines, 1.0% of training data
#   # ${V}.I8.Cardiomegaly.densenet121.baseline.fromscratch    python deepfix/train.py --deepfix off --epochs 150 --dset chexpert_small_ID:1000:$train_pct:.01 --opt Adam:lr=0.003 --lossfn chexpert_identity:$num_identities --loss_reg none --model densenet121:untrained:1:1
#   # ${V}.I8.leaderboard.densenet121.baseline.fromscratch    python deepfix/train.py --deepfix off --epochs 150 --dset chexpert_small_ID:1000:$train_pct:.01 --opt Adam:lr=0.003 --lossfn chexpert_identity:$num_identities --loss_reg none --model densenet121:untrained:1:5
#   # ${V}.I8.diagnostic.densenet121.baseline.fromscratch    python deepfix/train.py --deepfix off --epochs 150 --dset chexpert_small_ID:$num_identities:$train_pct:.01 --opt Adam:lr=0.003 --lossfn chexpert_identity:$num_identities --loss_reg none --model densenet121:untrained:1:$num_identities
#   cat <<EOF
#   # ${V}.I8.Cardiomegaly.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --epochs 150 --dset chexpert_small_ID:1000:$train_pct:.01 --opt Adam:lr=0.003 --lossfn chexpert_identity:$num_identities --loss_reg none --model resnet18:untrained:1:1
#   # ${V}.I8.leaderboard.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --epochs 150 --dset chexpert_small_ID:1000:$train_pct:.01 --opt Adam:lr=0.003 --lossfn chexpert_identity:$num_identities --loss_reg none --model resnet18:untrained:1:5
#   # ${V}.I8.diagnostic.resnet18.baseline.fromscratch    python deepfix/train.py --deepfix off --epochs 150 --dset chexpert_small_ID:$num_identities:$train_pct:.01 --opt Adam:lr=0.003 --lossfn chexpert_identity:$num_identities --loss_reg none --model resnet18:untrained:1:$num_identities
# EOF
# }
C9() {
  # experiment over varying patch_features. conc: l1 and min and max look good.
  python <<EOF
for levels, patch_size, patch_features in [
        (5,5,'l1V2'), (5,5,'sum'), (5,5,'l1'),
        (5,5,'max'), (5,5,'min'), (5,5,'median'),
        # (8,1,'l1'), (8,1,'l2'), (8,1,'sum_pos,sum_neg'),
        # (1,128,'l1'), (1,128,'l2'), (1,128,'sum_pos,sum_neg'),
        # (5,5,'l1'), (5,5,'l2'), (5,5,'sum_pos,sum_neg'), (4,8,'sum_pos,sum_neg'),
        ]:
    model = f'waveletmlpV2:1:14:coif1:{levels}:{patch_size}:{patch_features}'
    print(f""" ${V}.C9.{model}     python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model} """)
EOF
}
C10() {
  # experiment over varying wavelet types
  # conc: coif2, db1 and db2 look good, but not much difference in general
  python <<EOF
for wavelet in [
        # <= 6 filter size
        'bior1.1', 'bior1.3', 'bior2.2', 'bior3.1', 'coif1', 'coif2', 'coif3', 'db1', 'db2',
        'db3', 'rbio1.1', 'rbio1.3', 'rbio2.2', 'rbio3.1', 'sym2', 'sym3']:
    for levels, patch_size, patch_features in [(5,5,'l1'),]:
        model = f'waveletmlpV2:1:14:{wavelet}:{levels}:{patch_size}:{patch_features}'
        print(f""" ${V}.C10.{model}     python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model} """)
EOF
}
C11() {
  # experiment with deepfix_v2
  # conc: not working needs work.
  local model="deepfix_v2:1:14:coif1:5:5:l1:resnet18:imagenet"
        # lambda in_ch, out_ch, wavelet, wavelet_levels, patch_size, patch_features, backbone, pretraining: get_DeepFixEnd2End_v2(
  echo "${V}.C11.$model" python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model "$model"
  echo "${V}.C11.$model.lr001" python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model "$model"
  # python deepfix/train.py --dset chexpert_small:.01:.01:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model "$model"
}
C12() {
  # varying MLP depth and width,    all "s" and "m" have same num parameters, respectively
  # conc:  width=300, depth=1 hidden layer (+ vecattn layer) is just fine!
  python <<EOF
d = [2, 6, 10, 14, 18]
ws = [300, 174, 135, 114, 100]  # 180,000 parameter models
wm = [600, 347, 268, 226, 200]  # 4*180k parameter models
for depth, width in zip(d, ws):
    model = f'waveletmlp:{width}:1:14:5:5:1:{depth}'
    print(f""" ${V}.C12.s.{model}     python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model} """)
for depth, width in zip(d, wm):
    model = f'waveletmlp:{width}:1:14:5:5:1:{depth}'
    print(f""" ${V}.C12.m.{model}     python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model} """)
for depth, width in [(1, 600), (1, 300), (1, 1000), (0, 300)]:
    model = f'waveletmlp:{width}:1:14:5:5:1:{depth}'
    print(f""" ${V}.C12.1.{model}     python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model} """)
EOF
}

C13() {
  # normalization tests:  show that batchnorm improves perf (partially replicate elvin result).  does a global norm do the same?
  # mlp has depth 1 (+ vecattn layer) and width 300.
  # note: this includes CPU intensive tasks.
  # conc: the 0:0mean,chexpert_small normalization is best with l1.  with min, 0:whiten or 0:none or 0:0mean are all fine.
  # details:
  #   test 1: normalization after encoder
  #         l1:  none < batchnorm < whiten < 0mean
  #         min: batchnorm < (none == whiten == 0mean)  or on val set, 0mean is best
  #   test 2: normalization before encoder.
  #       l1:
  #         none:        0 =  1
  #         batchnorm:   0 ~> 1
  #         0mean:       0 = 1
  #         whiten:      0 ~>  1
  #       min:
  #         none:        0 = 1
  #         batchnorm:   0 = 1
  #         0mean:       0 ? 1
  #         whiten:      0 > 1
  #   test overall:
  #         l1 > min  (when using 0:0mean)
  python <<EOF
wavelet = "coif2"
level = 5
patchsize = 5
for zero_mean in '0', '1':
    for patch_features in ["l1", "min"]:
        print(f" python bin/compute_deepfix_normalization.py --wavelet {wavelet} --level {level} --patchsize {patchsize} --patch_features {patch_features} --zero_mean {zero_mean}")

        for normalization in ["none", "batchnorm", "whiten,chexpert_small", "0mean,chexpert_small"]:
            model = f"waveletmlp_bn:1:14:{wavelet}:{level}:{patchsize}:{patch_features}:{zero_mean}:{normalization}"
            print(f"""${V}.C13.{model} python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model} """)
EOF
}

compute_normalization() {
  python <<EOF
wavelet = 'db1'
patch_features = 'l1'
for dset in 'chexpert_small','chexpert':
    for zero_mean in '0': #, '1':
        for level in [1,5,8]:  #range(1, 9):
            # for patchsize in 1,3,5,9,19,37,79,115,160:
            for patchsize in 1,5,160:
        # for level in range(1, 9):
        #     for patchsize in 1,3,5,9,19,37,79,115,160:
                if patchsize <= 320 / 2**level:
                    # run_id:   norm:{dset}:{wavelet}:{level}:{patchsize}:{patch_features}:{zero_mean}
                    print(f"python bin/compute_deepfix_normalization.py --dset {dset} --wavelet {wavelet} --level {level} --patchsize {patchsize} --patch_features {patch_features} --zero_mean {zero_mean}")
                # else skip this unnecessary norm because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}


C14() {
  # varying attention / regularization
  # conc: perf-wise: VecAttn with 0 regualrization seems best, .1 reg 2nd best, LogSoftmaxVecAttn third
  # conc: but the result that vecattn:0 works better than comparable model in C13 is weird enough that might have optimizer noise here.
  # conc: visually for attn, TODO

  python <<EOF
level = 5
patch_size = 5
for attn in 'Identity', 'SoftmaxVecAttn', 'LogSoftmaxVecAttn':
    model = f'attn_test:{attn}:14:{level}:{patch_size}'
    print(f"""${V}.C14.{attn} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model {model}""")
model = f'attn_test:VecAttn:14:{level}:{patch_size}'
for reg in 0, .1, 1:
    print(f"""${V}.C14.VecAttn.{reg} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:{reg} --model {model}""")
EOF
}

C15() {
  # optimizer tests for two attn models
  python <<EOF
level = 5
patch_size = 5
reg = .1
for opt in 'SGD:lr=0.002:momentum=.9', 'SGD:lr=0.003:momentum=.9', 'SGD:lr=0.001:momentum=.9', 'SGD:lr=0.001', 'SGD:lr=0.0005', 'Adam:lr=0.001', 'Adam:lr=0.002', 'Adam:lr=0.005':
    print(f"""${V}.C15.VecAttn-{reg}.{opt}   python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt {opt} --lossfn chexpert_uignore --loss_reg deepfixmlp:{reg} --model attn_test:VecAttn:14:{level}:{patch_size}""")
    print(f"""${V}.C15.LogSoftmax.{opt}      python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt {opt} --lossfn chexpert_uignore --loss_reg none             --model attn_test:LogSoftmaxVecAttn:14:{level}:{patch_size}""")
EOF
}
# before running this, let's do the
# regularization tests
# optimizer tests

C16() {
  # Main predictive experiment, all patch sizes and wavelet levels
  # with coif2
  python <<EOF
for level in range(1, 9):
    for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
        #     print(f"norm:{level}:{patchsize}:{patch_features}:{zero_mean} python bin/compute_deepfix_normalization.py --level {level} --patchsize {patchsize} --patch_features {patch_features} --zero_mean {zero_mean}")
            model = f"deepfix_v1:14:{level}:{patchsize}"
            print( f"${V}.C16.J={level}.P={patchsize} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}
C17() {
  # Per-class model, to get hierarchy over classes
  python <<EOF
class_names = [
    'No Finding',
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices']
class_names = [x.replace(' ', '\ ') for x in class_names]
level = 5
patchsize = 5
for class_name in class_names:
    model = f"deepfix_v1:1:{level}:{patchsize}"
    class_name2 = class_name.replace('\ ', '_')
    print( f"""${V}.C17.{class_name2} python deepfix/train.py --dset chexpert_small15k:.9:.1:{class_name2} --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80 """)
EOF
}

C18() {
  # Adaptive=1
  # Main predictive experiment, all patch sizes and wavelet levels
  local V=$((V+2))
  # v=v  uses wavelet=db1 with standard 0mean normalization
  # v=v+1 same as above, but hacked to compute selu in each recursive level
  # v=v+2 uses wavelet='normal_:2' and normalization='batchnorm'
  python <<EOF
wavelet = 'normal_,2'
normalization = 'batchnorm'  #  '0mean,chexpert_small'
for level in [5]: # ,8,1]:  #range(1, 9):
    # for patchsize in 1,3,5,9,19,37,79,115,160:
    # for patchsize in 5,1,160:
    for patchsize in 5,:
        if patchsize <= 320 / 2**level:
            model = f"deepfix_v1:14:{level}:{patchsize}:1:{wavelet}:{normalization}"
            print( f"${V}.C18.J={level}.P={patchsize} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}
C19() {
  # Adaptive=2
  # Main predictive experiment, all patch sizes and wavelet levels
  # v=v  uses wavelet=db1
  # v=v+1 uses wavelet=db1, but hacked to compute selu in each recursive thing
  local V=$((V+1))
  python <<EOF
for level in [1,5,8]:  #range(1, 9):
    # for patchsize in 1,3,5,9,19,37,79,115,160:
    for patchsize in 1,5,160:
        if patchsize <= 320 / 2**level:
            model = f"deepfix_v1:14:{level}:{patchsize}:2"
            print( f"${V}.C19.J={level}.P={patchsize} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}
C20() {
  # Full size images
  python <<EOF
# for level in range(1, 9):
#     for patchsize in 1,3,5,9,19,37,79,115,160:
for level in [5,8,1]:  #range(1, 9):
    for patchsize in 5,160,1:
        if patchsize <= 320 / 2**level:
        #     print(f"norm:{level}:{patchsize}:{patch_features}:{zero_mean} python bin/compute_deepfix_normalization.py --level {level} --patchsize {patchsize} --patch_features {patch_features} --zero_mean {zero_mean}")
            model = f"deepfix_v1:14:{level}:{patchsize}"
            print( f"${V}.C20.J={level}.P={patchsize} python deepfix/train.py --dset chexpert15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80 --start_epoch 1")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}
C21() {
  # Main predictive experiment, all patch sizes and wavelet levels, trained on diagnostic classes
  # with db1 (since coif2 has large filters, and edge artifacts might be problematic)
  python <<EOF
# for level in [1,5,8]:
#     for patchsize in 1,5,160:
for level in range(1, 9):
    for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
        #     print(f"norm:{level}:{patchsize}:{patch_features}:{zero_mean} python bin/compute_deepfix_normalization.py --level {level} --patchsize {patchsize} --patch_features {patch_features} --zero_mean {zero_mean}")
            model = f"deepfix_v1:14:{level}:{patchsize}:0:db1"
            print( f"${V}.C21.J={level}.P={patchsize} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

C22()  {
  # Compute the re-identification privacy scores
  python <<EOF
wavelet = 'db1'
n_patients = 2000
# n_patients = 6  # for testing
# for level in [8,5,1]:
#     for patchsize in 1,5,160:
for level in range(1, 9):
    for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
            print(f'''${V}.C22.{n_patients}.{wavelet}.J={level}.P={patchsize} python bin/anonymity_score.py --n_bootstrap 6 --wavelet {wavelet} --level {level} --patchsize {patchsize} --n_patients {n_patients} --plot ''')
EOF
}

C23() {
  # Adapt, unet before encoder
  python <<EOF
for level in [5,8,1]:
    for patchsize in 5,1,160:
# for level in range(1, 9):
    # for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
            model = f"adeepfix_v1:14:{level}:{patchsize}"
            print( f"${V}.C23.J={level}.P={patchsize} python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

C24() {
  # Main predictive experiment, all patch sizes and wavelet levels, trained on only the 5 leaderboard classes
  # version V+0:  with db1, with deepfixmlp:1, with leaderboard classes
  # version V+1:  with db1, deepfixmlp:0, leaderboard
  local V=$((V+1))
  python <<EOF
# for level in [1,5,8]:
#     for patchsize in 1,5,160:
# hack: my gpu has a hardware problem.  too high batch size or num workers causes server to crash.  bypass it this way.
for level, patchsize in [(1,160), (6,5), (3,19), (8,1)]:
    model = f"deepfix_v1:5:{level}:{patchsize}:0:db1"
    print( f"${V}.C24.J={level}.P={patchsize} env num_workers=0 batch_size=200 python deepfix/train.py --dset chexpert_small15k:.9:.1:leaderboard --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:0 --model {model} --epochs 80")
for level in range(1, 9):
    for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
        #     print(f"norm:{level}:{patchsize}:{patch_features}:{zero_mean} python bin/compute_deepfix_normalization.py --level {level} --patchsize {patchsize} --patch_features {patch_features} --zero_mean {zero_mean}")
            model = f"deepfix_v1:5:{level}:{patchsize}:0:db1"
            print( f"${V}.C24.J={level}.P={patchsize} env num_workers=5 batch_size=600 python deepfix/train.py --dset chexpert_small15k:.9:.1:leaderboard --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:0 --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

C25() {
  # Main predictive experiment, DeepFixImg2Img --> DenseNet121
  # repeat of C21
  python <<EOF
# for level in [1,5,8]:
#     for patchsize in 1,5,160:
for level in range(1, 9):
    for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
            model = f"deepfix_densenet121:1:{level}:{patchsize}:{patchsize}"
            print( f"${V}.C25.J={level}.P={patchsize} env num_workers=6 batch_size=50 python deepfix/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --model {model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

timings_e2e_table() {
  # data for the end-to-end timing table.
python <<EOF
for J,P,batch_size in [ (6,5,1000), (2,19,1000),(1,1,1350), (1,160,800), (8,1,600), (5,5,1350) ]:
    model = f"deepfix_v1:14:{J}:{P}:0:db1"
    print(f'''${V}.timing.DeepFix.J={J}.P={P}  env batch_size={batch_size} num_workers=6  python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg deepfixmlp:.1 --model {model} --epochs 6 --start_epoch 1''')
EOF
  cat <<EOF
${V}.timing.ResNet18                           env batch_size=55 num_workers=6    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model resnet18:untrained:1:14 --epochs 6 --start_epoch 1
${V}.timing.Densenet121                        env batch_size=10 num_workers=6    python deepfix/train.py --deepfix off --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model densenet121:untrained:1:14 --epochs 6 --start_epoch 1
EOF
}

timings_2_table() {
  # data for the timings table for encoder
  python<<EOF
for J,P in [(6,5), (0,0), (2,19), (1,1), (1,160)]:
  for dev in ['cpu','cuda']:
    print(f"""${V}.timing2.{J}.{P}.{dev} env batch_size=1000 num_workers=10 python bin/table_timing_encoder_decoder.py -J {J} -P {P} --device {dev}""")
EOF
}

filesizes_table() {
  # data to report on-disk compression performance
python <<EOF
for J,P in [(6,5), (0,0), (2,19), (1,1), (1,160)]:
  for fmt in ['jpg','png']:
    print(f"""python bin/make_compressed_dataset.py -J {J} -P {P} --read_from_dirpath ./data/CheXpert-v1.0-small/  --compressed_img_format {fmt} --dsets valid""")
for J,P in [(6,5), (0,0), (2,19), (1,1), (1,160)]:
  for fmt in ['jpg','png']:
    print(f"""echo {J} {P} {fmt} \$(find data/CheXpert-v1.0-small.compressed_{fmt}.J={J}.P={P}/valid -type f| parallel du -b |  datamash mean 1)""")
# the train set
for fmt in ['jpg','png']:
  print(f"""python bin/make_compressed_dataset.py -J 2 -P 19 --read_from_dirpath ./data/CheXpert-v1.0-small/  --compressed_img_format {fmt} --dsets train""")
EOF
}

plots() {
  # compression ratio
  python ./bin/plot_compression_ratio.py --patch_sizes 1 3 5 9 19 37 79 115 160
  python ./bin/plot_compression_ratio.py --patch_sizes 1 2 3 4 5 6 7 8 --input_size 64 64

  # predictive performance (after running C21 (or C16))
  # C21 | run_gpus 1
  # C8 | run_gpus 1
  # --> get ROC AUC and BAcc data on test set by loading saved model checkpoints
  python bin/get_roc_auc.py 3.C8 --overwrite
  python bin/get_roc_auc.py 2.C21 --overwrite
  python bin/get_roc_auc.py 2.C24 --overwrite
  # --> make the heatmap plot
  bin/plot_perf_rocauc_heatmap.py 2.C21
  bin/plot_perf_rocauc_heatmap.py 2.C24
  # --> BAcc on val set, with 0.5 threshold instead of optimal threshold
  # ./bin/plot_heatmap_levels_vs_patchsize.sh 2.C21

  # privacy: re-identification  (after running C22)
  # C22 | run_gpus 1
  ./bin/plot_ks_heatmap.py 2000 1
  # --> reconstruction heatmap plot (ssim)
  batch_size=300 python ./bin/plot_ssim_heatmap.py --overwrite
  batch_size=300 python ./bin/plot_ssim_heatmap.py --overwrite --patch_features sum
  # --> pictures of reconstructed images
  python ./bin/plot_reconstructions.py
  python ./bin/plot_reconstructions.py --ssim
  # ... and appendix
  python ./bin/plot_reconstructions.py --patch_features sum
  python ./bin/plot_reconstructions.py --ssim --patch_features sum

  # visualize the 4-d cube with scatter matrix (after running above plots)
  # not used
  # python ./bin/plot_3d.py

  # make a privatized and compressed chexpert dataset on the disk
  # python bin/make_compressed_dataset.py -J 2 -P 19 --centercrop 320 320 --compressed_img_format png --read_from_dirpath ./data/CheXpert-v1.0-small/
}

plots_intelmobileodt() {
    local patch_sizes="1 5 7 11 31 75"
    # local patch_sizes="1 3 5 7 9 11 21 31 41 51 61 71 75"

python ./bin/plot_reconstructions.py --dataset intelmobileodt --patch_sizes $patch_sizes
./bin/plot_heatmap_levels_vs_patchsize.sh 2.E6 $patch_sizes
python ./bin/plot_compression_ratio.py --patch_sizes $patch_sizes --input_shape 200 150
python bin/plot_ssim_heatmap.py --dataset intelmobileodt --overwrite --patch_sizes $patch_sizes
# TODO: identity score
}


# I1 | expand 3 | run_gpus echo 5
# I2 | run_gpus 5
# I3_part1 | parallel -j 5
# I3_part2 I3_part2 I3_part2 | parallel -j 5
# I3_part2 | run_gpus 5
# I4 | parallel -j 1

# E1 | parallel -j 1
# for i in {1..6} ; do
# E2 | run_gpus 3
# E3 | run_gpus 3
# E4 | run_gpus 3
# E5 | run_gpus 1
# C3 | run_gpus 3
# ( C5; C4 ) | run_gpus 4
# C6_vary_patch_size | run_gpus 3
# C6_vary_patch_size_part2 | run_gpus 1
# done
# C7 | run_gpus 4
# ( I8; C8 ) | run_gpus 3
# C9 #| run_gpus 3
# export num_workers=4
# export batch_size=15
# C8 | run_gpus 3
# ( C11 ; C8 ) | run_gpus 3
# ( C9 ; C10 ) | run_gpus 5
# ( C8 ; C9 ; C10 ; C11 ; C12 ) | run_gpus 3
# C12
# C13 | grep compute_deepfix | parallel -j 10
# C13 | grep -v compute_deepfix | run_gpus 5
# C15 | run_gpus 1
# export num_workers=6
# export batch_size=16
# compute_normalization | grep -v chexpert_small | parallel -j 1  # "{} --device cpu"
# C20 | run_gpus 3  # 4.588 gb gpu ram for batchsize=10
# export num_workers=2
# export batch_size=200
# compute_normalization | grep chexpert_small | parallel -j 5
# # ( C17 ) #| run_gpus 1
# ( C16 ; C17 ; C18 ) | run_gpus 1
# C19 | run_gpus 1
# C21 | run_gpus 1
# C22 | run_gpus 1
# export num_workers=0
# export batch_size=400
# C23 | run_gpus 1
# C18 | run_gpus 1
# export batch_size=100
# C19 | run_gpus 1
# timings_e2e_table | run_gpus 1
# timings_2_table | run_gpus 1
# filesizes_table | parallel -j 1
# C24 | run_gpus 1
# C8 | run_gpus 1

# C25 | run_gpus 2

E6() {
  # Main predictive experiment, all patch sizes and wavelet levels, trained on only the 5 leaderboard classes
  python <<EOF
args = ' --opt Adam:lr=0.001 --epochs 300 --dset intel_mobileodt:train+additional:val:test:v1 --lossfn CrossEntropyLoss '
for level in [1,2,3,4,5,6,7]:
    for patchsize in 1,3,5,7,9,11,21,31,41,51,61,75:
        if patchsize <= 200 / 2**level:
            # model = f"deepfix_cervical:{level}:{patchsize}"
            model = f"deepfix_resnet18:3:3:{level}:{patchsize}:{patchsize}"
            print( f"""${V}.E6.J={level}.P={patchsize} env num_workers=12 batch_size=6000 python deepfix/train.py --model {model} {args}""")
EOF
}
E7() {
args=' --opt Adam:lr=0.001 --epochs 300 --dset intel_mobileodt:train+additional:val:test:v1 --lossfn CrossEntropyLoss '
    cat <<EOF
${V}.E7.resnet18 env num_workers=12 batch_size=6000 python deepfix/train.py --dset intel_mobileodt:train+additional:val:test:v1 --model resnet18:untrained:3:3  $args
EOF
# env num_workers=0 batch_size=2600 python deepfix/train.py --model densenet121:untrained:3:3 --opt SGD:lr=0.001 --epochs 300 --dset intel_mobileodt:train:val:test:v1 --lossfn CrossEntropyLoss  --loss_reg none --start_epoch 1
}


K0() {
    cat <<EOF
$V.K0 env num_workers=10 batch_size=200  python deepfix/train.py --model resnet18:untrained:3:3 --opt Adam:lr=0.001 --epochs 50 --dset kimeye:.7:.15 --lossfn kimeye_ce
EOF
}

K1() {
    # Main experiment for Kim's Eye Hospital Glaucoma dataset
  python <<EOF
args = ' --opt Adam:lr=0.001 --epochs 50 --dset kimeye:.7:.15 --lossfn kimeye_ce '
for level in [1,2,3,4,5,6,7]:
    for patchsize in 1,3,5,7,9,11,21,31,41,51,61,75:
        if patchsize <= 200 / 2**level:
            model = f"deepfix_resnet18:3:3:{level}:{patchsize}:{patchsize}"
            print( f"""${V}.K1.J={level}.P={patchsize} env num_workers=10 batch_size=200   python deepfix/train.py --model {model} {args}""")
EOF
}

plots_kimeye() {
    local patch_sizes="1 3 5 7 9 11 15 16 17 21 31 51 75 101"
    # local patch_sizes="1 3 5 7 9 11 21 31 41 51 61 71 75"

    # todo: fit the
./bin/plot_heatmap_levels_vs_patchsize.sh 2.K1 $patch_sizes
python ./bin/plot_compression_ratio.py --patch_sizes $patch_sizes --input_shape 200 150
python ./bin/plot_reconstructions.py --dataset kimeye --patch_sizes $patch_sizes
python bin/plot_ssim_heatmap.py --dataset kimeye --overwrite --patch_sizes $patch_sizes
}

# E6 | run_gpus 2
# # E3 | run_gpus 2
export lockfile_maxsuccesses=6
( E7 ; E7 ; E7 ; E7 ; E7 ; E7 ) | run_gpus 2
# plots_intelmobileodt

( K1 ; K1 ; K1 ; K1 ; K1 ; K1 ; K0 ; K0 ; K0 ; K0 ; K0 ; K0 ) | run_gpus 1
# plots_kimeye
