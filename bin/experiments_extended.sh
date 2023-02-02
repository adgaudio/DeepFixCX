#!/usr/bin/env bash
# a script to reproduce our experiments.

. ./bin/activate
set -e
set -u
# set -o pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"

. ./bin/bash_lib.sh

lockfile_maxsuccesses=1
lockfile_maxconcurrent=1
lockfile_maxfailures=1

V=2  # experiment version number


C8_baselines() {
    # CheXpert baselines
  local V=$((V+2))
    cat <<EOF
${V}.C8.diagnostic.volo_d1_224.baseline env num_workers=11 batch_size=10 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt AdamW:lr=0.001:amsgrad=1 --lossfn chexpert_uignore --loss_reg none --model volo_d1_224:1:14 --epochs 80
${V}.C8.diagnostic.efficientnetv2_m.baseline env num_workers=11 batch_size=15 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model efficientnetv2_m:1:14 --epochs 80
${V}.C8.diagnostic.efficientnet-b0.baseline env num_workers=11 batch_size=50 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model efficientnet-b0:imagenet:1:14 --epochs 80
${V}.C8.diagnostic.vip_s7.baseline env num_workers=11 batch_size=60 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model vip_s7:1:14 --epochs 80
${V}.C8.diagnostic.mdmlp_320.baseline env num_workers=11 batch_size=100 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model mdmlp_320:1:14 --epochs 80
${V}.C8.diagnostic.coatnet_1_224.baseline.adamw2 env num_workers=11 batch_size=25 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt AdamW:lr=0.001:weight_decay=.0000001:amsgrad=1 --lossfn chexpert_uignore --loss_reg none --model coatnet_1_224:1:14 --epochs 80
EOF

}

generate_experiment_deepfixcx_chexpert() {
    local experiment_id="$1" ; shift
    local deepfixcx_model="$1" ; shift
    local opt="${1}" ; shift
    local num_workers="$1" ; shift
    local batch_size="$1" ; shift
    python <<EOF
for level in range(1, 9):
    for patchsize in 1,3,5,9,19,37,79,115,160:
        if patchsize <= 320 / 2**level:
            model = f"${deepfixcx_model}"
            print( f"${V}.${experiment_id}.J={level}.P={patchsize} env num_workers=$num_workers batch_size=$batch_size python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt ${opt} --lossfn chexpert_uignore --model ${deepfixcx_model} --epochs 80")
        # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

C29() {
  # repeat C28, with volo instead of densenet
  # chexpert dataset
  generate_experiment_deepfixcx_chexpert C29 "deepfixcx_volo_d1_224:1:14:{level}:{patchsize}:{patchsize}" "AdamW:lr=0.001:amsgrad=1" 11 60
}

C30() {
  # repeat C28, with efficientnet-b0 instead of densenet
  # chexpert dataset
    generate_experiment_deepfixcx_chexpert C30 "deepfixcx_efficientnet-b0:1:14:{level}:{patchsize}:{patchsize}" "Adam:lr=0.001" 11 50
}

C31() {
    # Gaussian Blur of 320x320 chexpert images
    python <<EOF
for kernel_size in 4,8,16,32,64:  #,128,160,192,224,256,288:
        print( f"${V}.C31.K={kernel_size} env num_workers=11 batch_size=15 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --model blur_efficientnet-b0:1:14:{kernel_size} --epochs 80")
            # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

C32() {
    # Median Pooling of 320x320 chexpert images
    python <<EOF
for kernel_size in 3,5,9,15,25,50,75:
        print( f"${V}.C32.K={kernel_size} env num_workers=11 batch_size=50 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --model medianpool2d_efficientnet-b0:1:14:{kernel_size} --epochs 80")
            # # else skip this unnecessary task because the (level, patchsize) isn't doing compression.  This assumes images are 320x320, our default from chexpert dataset
EOF
}

C33() {
    # related methods.  spot checking deepfixcx-DNN versus DNN at a particular J and P
    local J=1
    local P=115
    cat<<EOF
${V}.C33.J=${J}.P=${P}.efficientnetv2_m env num_workers=11 batch_size=15 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model deepfixcx_efficientnetv2_m:1:14:${J}:${P}:${P} --epochs 80
${V}.C33.J=${J}.P=${P}.resnet18   env num_workers=4 batch_size=60 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.003 --lossfn chexpert_uignore --loss_reg none --model deepfixcx_resnet18:imagenet:1:14:${J}:${P}:${P} --epochs 80
${V}.C33.J=${J}.P=${P}.vip_s7 env num_workers=11 batch_size=60 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model deepfixcx_vip_s7:1:14:${J}:${P}:${P}  --epochs 80
EOF
    local J=1
    local P=160
    cat <<EOF
${V}.C33.J=${J}.P=${P}.efficientnetv2_m env num_workers=11 batch_size=15 python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt Adam:lr=0.001 --lossfn chexpert_uignore --loss_reg none --model deepfixcx_efficientnetv2_m:1:14:${J}:${P}:${P} --epochs 80
EOF
}

C31b() {
    # related methods
    # Gaussian Blur and Median: evaluate closest deepfixcx models to:
    # - blur tests 4,8,16,32,64
    # - median tests 3,5,9,15,25,50,75
    # Lucky that median and gaussian are basically the same (J,P) ==> less tests :)
    # NOTE:  The closest deepfixcx (J,P) parameters were found by the function
    #        "find_deepfixcx_model_most_comparable_to(...)"
    #        in file plot_competing_methods_rocauc.py
    local deepfixcx_model='deepfixcx_efficientnet-b0:1:14:{J}:{P}:{P}'
    local experiment_id='C31b'
    local opt="Adam:lr=0.001"
    local num_workers=11
    local batch_size=50
    python <<EOF
J=1
blur_closest_P = 5,10,20,40,80
median_closest_P = 80,80,40,40,18,6,4
for P in set(blur_closest_P + median_closest_P):
    model = f"${deepfixcx_model}"
    print( f"${V}.${experiment_id}.J={J}.P={P} env num_workers=$num_workers batch_size=$batch_size python deepfixcx/train.py --dset chexpert_small15k:.9:.1:diagnostic --opt ${opt} --lossfn chexpert_uignore --model ${deepfixcx_model} --epochs 80")
EOF
}

C35() {
    # C28 experiment with deepfixcx_mdmlp
    generate_experiment_deepfixcx_chexpert C35 "deepfixcx_mdmlp_320:1:14:{level}:{patchsize}:{patchsize}" "Adam:lr=0.003" 11 100
}

C36() {
    # C28 experiment with deepfixcx_coatnet_1_224
    # comparing to the adamw2 baseline
    generate_experiment_deepfixcx_chexpert C36 "deepfixcx_coatnet_1_224:1:14:{level}:{patchsize}:{patchsize}" "AdamW:lr=0.001:weight_decay=.0000001:amsgrad=1" 11 20
}

F1() {
    # Flowers 102 dataset, baselines
    cat <<EOF
${V}.F1.diagnostic.mdmlp_flowers102.baseline env num_workers=11 batch_size=35 python deepfixcx/train.py --dset flowers102 --opt AdamW:lr=0.001:weight_decay=0.0001:amsgrad=1 --lossfn CrossEntropyLoss --loss_reg none --model mdmlp_patch14_lap7_dim64_depth8_224:3:102 --epochs 200
EOF
}
generate_experiment_deepfixcx_flowers102() {
    local experiment_id="$1" ; shift
    local deepfixcx_model="$1" ; shift
    local opt="${1}" ; shift
    local num_workers="$1" ; shift
    local batch_size="$1" ; shift
    python <<EOF
for level in range(1, 8):
    for patchsize in 1,23,45,67,78,89,111:
        if patchsize <= 224 / 2**level:
            model = f"${deepfixcx_model}"
            print( f"${V}.${experiment_id}.J={level}.P={patchsize} env num_workers=$num_workers batch_size=$batch_size python deepfixcx/train.py --dset flowers102 --opt ${opt} --lossfn CrossEntropyLoss --model ${deepfixcx_model} --epochs 200")
EOF
}
F2() {
    # DeepFixCX Flowers102 experiment: mdmlp
    # TODO: update this after finalize choices for F1
    generate_experiment_deepfixcx_flowers102 F2 "deepfixcx_mdmlp_patch14_lap7_dim64_depth8_224:3:102:{level}:{patchsize}:{patchsize}" AdamW:lr=0.001:weight_decay=0.0001:amsgrad=1 11 30
}

G1() {
    # Food101 Dataset, baseline
    cat <<EOF
${V}.G1.diagnostic.mdmlp_food101.baseline env num_workers=11 batch_size=35 python deepfixcx/train.py --dset food101 --opt AdamW:lr=0.001:weight_decay=0.000001:amsgrad=1 --lossfn CrossEntropyLoss --loss_reg none --model mdmlp_patch14_lap7_dim64_depth8_224:3:101 --epochs 60
EOF
}
G2() {
    local experiment_id="G2"
    local deepfixcx_model="deepfixcx_mdmlp_patch14_lap7_dim64_depth8_224:3:101:{level}:{patchsize}:{patchsize} "
    local opt="AdamW:lr=0.001:weight_decay=0.000001:amsgrad=1"
    local num_workers="11"
    local batch_size="35"
    local dset="food101"
    python <<EOF
for level in range(1, 8):
    for patchsize in 1,23,45,67,89,111:
        if patchsize <= 224 / 2**level:
            model = f"${deepfixcx_model}"
            print( f"${V}.${experiment_id}.J={level}.P={patchsize} env num_workers=$num_workers batch_size=$batch_size python deepfixcx/train.py --dset $dset --opt ${opt} --lossfn CrossEntropyLoss --model ${deepfixcx_model} --epochs 60")
EOF
}
G3() {
    local experiment_id="G3"
    local deepfixcx_model="mdmlp_patch14_lap7_dim64_depth8_224:3:101"
    local opt="AdamW:lr=0.001:weight_decay=0.000001:amsgrad=1"
    local num_workers="0"
    local batch_size="35"
    local dset="food101:{level}:{patchsize}"
    python <<EOF
for level in range(1, 9):
    for patchsize in 256,128,64,32,16,8,4,2:
        if patchsize <= 500 / 2**level:
            print( f"${V}.${experiment_id}.J={level}.P={patchsize} env num_workers=$num_workers batch_size=$batch_size python deepfixcx/train.py --dset $dset --opt ${opt} --lossfn CrossEntropyLoss --model ${deepfixcx_model} --epochs 40")
EOF
}


plots_extended() {
    # chexpert extended results
    chexpert=true ./bin/table_best_accuracy.sh 4.C8  # get the baseline numbers used below
    chexpert=true ./bin/plot_heatmap_levels_vs_patchsize.sh 2.C29 0.805294  # volo
    chexpert=true ./bin/plot_heatmap_levels_vs_patchsize.sh 2.C30 .87397  # efficientnet-b0
    # C31 C32 blur and median
    # C31b closest deepfixcx to blur/median
    # 34  doesn't exist (it's C31b, testing
    chexpert=true ./bin/plot_heatmap_levels_vs_patchsize.sh 2.C35 .8310  # mdmlp
    chexpert=true ./bin/plot_heatmap_levels_vs_patchsize.sh 2.C36 .828  # coatnet
    # chexpert=true ./bin/table_best_accuracy.sh 2.C31  #  blur
    # chexpert=true ./bin/table_best_accuracy.sh 2.C32  #  median
    # chexpert=true ./bin/table_best_accuracy.sh 2.C31b  #  blur and median: closest # deepfixcx
    # chexpert=true ./bin/table_best_accuracy.sh 2.C33  #  DeepFixCX+DNN results not covered by the other experiments
    python ./bin/plot_competing_methods.py  # updated based on outputs of table_best_accuracy.sh

    # privacy scores - blur and median plots

    # compression ratios:
    python bin/plot_compression_ratio.py --input_shape 224 224 --patch_sizes 1 23 45 67 89 111 --filenameid _food101
    python bin/plot_compression_ratio.py --input_shape 224 224 --patch_sizes 1 23 45 67 78 89 111 --filenameid _flowers102
    # ... chexpert 224x224 models still apply deepfixcx on the 320x320 images.
    #

    # Flowers and Food datasets:
    # ./bin/table_best_accuracy.sh 2.F2
    # ./bin/table_best_accuracy.sh 2.F1
    # food101=true ./bin/table_best_accuracy.sh 2.G2
    # food101=true ./bin/table_best_accuracy.sh 2.G1
    ./bin/plot_heatmap_levels_vs_patchsize.sh 2.F2 .528  # mdmlp Flowers
    food101=true ./bin/plot_heatmap_levels_vs_patchsize.sh 2.G2 .658  # mdmlp Food

    python ./bin/plot_ssim_heatmap.py --dataset food101 --patch_sizes 1 23 45 67 78 89 111 --overwrite
    python ./bin/plot_ssim_heatmap.py --dataset flowers102 --patch_sizes 1 23 45 67 78 89 111 --overwrite

    python ./bin/plot_reconstructions.py --dataset flowers102 --patch_sizes 1 23 45 67 78 89 111
    python ./bin/plot_reconstructions.py --dataset flowers102 --ssim  --patch_sizes 1 23 45 67 78 89 111

    python ./bin/plot_reconstructions.py --dataset food101 --patch_sizes 1 23 45 67 89 111
    python ./bin/plot_reconstructions.py --dataset food101 --ssim  --patch_sizes 1 23 45 67 89 111

    python ./bin/plot_reconstructions.py --dataset food101 --dataset food101_512 --patch_sizes 2 4 8 16 32 64 128 256
    python ./bin/plot_reconstructions.py --dataset food101 --dataset food101_512  --ssim --patch_sizes 2 4 8 16 32 64 128 256
}

# CheXpert
# ( C8_baselines ; C29 ; C30 ; C31 ; C32 ; C33 ; C31b ; C35 ; C36 ; F1 ; F2 ; G1 ; G2 ) | run_gpus 1
( C8_baselines ; C31b ; C35 ; C36 ; F1 ; F2 ; C33 ; G1 ; G2 ; G3 ) | run_gpus 1

# TODO: choose whether coatnet with adamw is preferred over adam.  make C36

# plots_extended
