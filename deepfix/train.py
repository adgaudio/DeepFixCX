"""
Training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets.
"""
#  import json
import os
import pampy
from simple_parsing import ArgumentParser
from simplepytorch import trainlib as TL
from simplepytorch import metrics
from typing import Union
import dataclasses as dc
import torch as T
import timm

from deepfix.models import (
    vip, MedianPool2d,
    get_effnetv2, get_resnet, get_densenet, get_efficientnetv1,
    get_DeepFixEnd2End, get_DeepFixEnd2End_v2, DeepFixMLP, UnetD,
    DeepFixImg2Img, MDMLP  # note: timm.create_model is also used.
)
from torchvision.transforms import GaussianBlur
from deepfix.dsets import (
    get_dset_chexpert, get_dset_intel_mobileodt, get_dset_kimeye,
    get_dset_flowers102, get_dset_food101, get_dset_food101_deepfixed)


def parse_normalization(normalization, wavelet, wavelet_levels, wavelet_patch_size, patch_features, zero_mean):
    if normalization.endswith(',chexpert_small'):
        fp = f'norms/chexpert_small:{wavelet}:{wavelet_levels}:{wavelet_patch_size}:{patch_features}:{zero_mean}.pth'
        if normalization == 'whiten,chexpert_small':
            return ('whiten', fp)
        elif normalization == '0mean,chexpert_small':
            return ('0mean', fp)
        else:
            raise NotImplementedError(normalization)
    else:
        return (normalization, )


MODELS = {
    ('effnetv2', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_effnetv2(pretrain, int(in_ch), int(out_ch))),
    ('resnet50', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_resnet('resnet50', pretrain, int(in_ch), int(out_ch))),
    ('resnet18', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_resnet('resnet18', pretrain, int(in_ch), int(out_ch))),
    ('densenet121', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_densenet('densenet121', pretrain, int(in_ch), int(out_ch))),
    ('efficientnet-b0', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_efficientnetv1('efficientnet-b0', pretrain, int(in_ch), int(out_ch))),
    ('efficientnet-b1', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_efficientnetv1('efficientnet-b1', pretrain, int(in_ch), int(out_ch))),
    ('waveletmlp', str, str, str, str, str, str, str): (
        lambda mlp_channels, in_ch, out_ch, wavelet_levels, patch_size, in_ch_mul, mlp_depth: get_DeepFixEnd2End(
            int(in_ch), int(out_ch),
            in_ch_multiplier=int(in_ch_mul), wavelet='db1',
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            mlp_depth=int(mlp_depth), mlp_channels=int(mlp_channels),
            mlp_fix_weights='none', mlp_activation=None,
            normalization=('none', ), mlp_attn='VecAttn')
        ),
    ('waveletmlpV2', str, str, str, str, str, str): (
        lambda in_ch, out_ch, wavelet, wavelet_levels, patch_size, patch_features: get_DeepFixEnd2End(
            int(in_ch), int(out_ch),
            in_ch_multiplier=1, wavelet=wavelet,
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            mlp_depth=2, mlp_channels=300,
            mlp_fix_weights='none', mlp_activation=None,
            patch_features=patch_features,
            normalization=('none', ), mlp_attn='VecAttn')
        ),
    ('waveletmlp_bn', str, str, str, str, str, str, str, str): (
        lambda in_ch, out_ch, wavelet, wavelet_levels, patch_size, patch_features, zero_mean, normalization: get_DeepFixEnd2End(
            int(in_ch), int(out_ch),
            in_ch_multiplier=1, wavelet=wavelet,
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            mlp_depth=1, mlp_channels=300,
            mlp_fix_weights='none', mlp_activation=None,
            patch_features=patch_features,
            zero_mean=bool(int(zero_mean)),
            normalization=parse_normalization(normalization, wavelet, wavelet_levels, patch_size, patch_features, zero_mean),
            mlp_attn='VecAttn', )
        ),
    ('attn_test', str, str, str, str): (
        lambda attn, out_ch, wavelet_levels, patch_size: get_DeepFixEnd2End(
            1, int(out_ch), in_ch_multiplier=1, wavelet='coif2',
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            patch_features='l1',
            mlp_depth=1, mlp_channels=300, mlp_fix_weights='none', mlp_activation=None,
            zero_mean=False, normalization=parse_normalization('0mean,chexpert_small', 'coif2', wavelet_levels, patch_size, 'l1', '0'),
            mlp_attn=attn,)
    ),
    ('deepfix_v1', str, str, str): (
        lambda out_ch, wavelet_levels, patch_size: get_DeepFixEnd2End(
            1, int(out_ch), in_ch_multiplier=1, wavelet='coif2',
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            patch_features='l1',
            mlp_depth=1, mlp_channels=300, mlp_fix_weights='none', mlp_activation=None,
            mlp_attn='VecAttn',
            zero_mean=False, normalization=parse_normalization('0mean,chexpert_small', 'coif2', wavelet_levels, patch_size, 'l1', '0'))
    ),
    # adaptive, placing unet before deepfix encoder
    ('adeepfix_v1', str, str, str): (
        lambda out_ch, wavelet_levels, patch_size: T.nn.Sequential(
            UnetD(channels=(1,3,6,12,24,48,96), depthwise_channel_multiplier=4,),
            get_DeepFixEnd2End(
                1, int(out_ch), in_ch_multiplier=1, wavelet='db1',
                wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
                patch_features='l1',
                mlp_depth=1, mlp_channels=300, mlp_fix_weights='none', mlp_activation=None,
                mlp_attn='VecAttn',
                zero_mean=False, normalization=parse_normalization('0mean,chexpert_small', 'db1', wavelet_levels, patch_size, 'l1', '0'))
        )
    ),
    # adaptive wavelet packet version:
    ('deepfix_v1', str, str, str, str): (
        lambda out_ch, wavelet_levels, patch_size, adaptive: get_DeepFixEnd2End(
            1, int(out_ch), in_ch_multiplier=1, wavelet='coif2',
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            patch_features='l1',
            mlp_depth=1, mlp_channels=300, mlp_fix_weights='none', mlp_activation=None,
            mlp_attn='VecAttn',
            zero_mean=False, normalization=parse_normalization('0mean,chexpert_small', 'coif2', wavelet_levels, patch_size, 'l1', '0'),
            adaptive=int(adaptive)
        )
    ),
    # adaptive wavelet packet version varying wavelet initialization:
    ('deepfix_v1', str, str, str, str, str): (
        lambda out_ch, wavelet_levels, patch_size, adaptive, wavelet: get_DeepFixEnd2End(
            1, int(out_ch), in_ch_multiplier=1, wavelet=wavelet,
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            patch_features='l1',
            mlp_depth=1, mlp_channels=300, mlp_fix_weights='none', mlp_activation=None,
            mlp_attn='VecAttn',
            zero_mean=False, normalization=parse_normalization('0mean,chexpert_small', wavelet, wavelet_levels, patch_size, 'l1', '0'),
            adaptive=int(adaptive)
        )
    ),
    # adaptive wavelet packet, supporting adaptive=1 or adaptive=2, different wavelets (including pytorch init like 'normal_:2'), and varying normalization
    ('deepfix_v1', str, str, str, str, str, str): (
        lambda out_ch, wavelet_levels, patch_size, adaptive, wavelet, normalization: get_DeepFixEnd2End(
            1, int(out_ch), in_ch_multiplier=1, wavelet=wavelet,
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            patch_features='l1',
            mlp_depth=1, mlp_channels=300, mlp_fix_weights='none', mlp_activation=None,
            mlp_attn='VecAttn',
            zero_mean=False, normalization=parse_normalization(normalization, wavelet, wavelet_levels, patch_size, 'l1', '0'),
            adaptive=int(adaptive)
        )
    ),
    # adaptive wavelet packet, supporting pytorch initialization and custom normalization
    ('deepfix_v2', str, str, str, str, str, str, str, str): (
        lambda in_ch, out_ch, wavelet, wavelet_levels, patch_size, patch_features, backbone, pretraining: get_DeepFixEnd2End_v2(
            int(in_ch), int(out_ch),
            in_ch_multiplier=1, wavelet=wavelet,
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            patch_features=patch_features,
            backbone=backbone, backbone_pretraining=pretraining,)
        ),
    ('deepfix_cervical', str, str): (lambda J, P: get_DeepFixEnd2End(
            in_channels=3, out_channels=3, in_ch_multiplier=1, wavelet='db1',
            wavelet_levels=int(J), wavelet_patch_size=int(P), patch_features='l1',
            mlp_depth=2, mlp_channels=300, mlp_activation=None,
            mlp_fix_weights='none',
            zero_mean=False, normalization=('batchnorm', ), mlp_attn='Identity',
            adaptive=0)),

    ('deepfix_resnet18', str, str, str, str, str): (lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
        DeepFixImg2Img(in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                       wavelet='db1', patch_features='l1',
                       restore_orig_size=False,
                       ),
        get_resnet('resnet18', 'untrained', int(in_ch), int(out_ch)),
    )),
    ('deepfix_densenet121', str, str, str, str, str): (lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
        DeepFixImg2Img(in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                       wavelet='db1', patch_features='l1',
                       restore_orig_size=False, min_size=(64,64)
                       ),
        get_densenet('densenet121', 'untrained', int(in_ch), int(out_ch)),
    )),

    ('volo_d1_384', str, str): (lambda in_ch, out_ch: T.nn.Sequential(
        T.nn.UpsamplingNearest2d((384, 384)),
        timm.create_model(
            'volo_d1_384',
            in_chans=int(in_ch), num_classes=int(out_ch), pretrained=True),
    )),
    ('volo_d1_224', str, str): (lambda in_ch, out_ch: T.nn.Sequential(
        T.nn.UpsamplingNearest2d((224, 224)),
        timm.create_model(
            'volo_d1_224',
            in_chans=int(in_ch), num_classes=int(out_ch), pretrained=True),
    )),
    ('deepfix_volo_d1_224', str, str, str, str, str): (lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
        DeepFixImg2Img(in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                       wavelet='db1', patch_features='l1',
                       restore_orig_size=False,
                       ),
        T.nn.UpsamplingNearest2d((224, 224)),
        timm.create_model(
            'volo_d1_224',
            in_chans=int(in_ch), num_classes=int(out_ch), pretrained=True),
    )),

    ('efficientnetv2_m', str, str): (lambda in_ch, out_ch: T.nn.Sequential(
        timm.create_model(
            'efficientnetv2_m',
            in_chans=int(in_ch), num_classes=int(out_ch), pretrained=False),
    )),
    ('deepfix_efficientnetv2_m', str, str, str, str, str): (lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
        DeepFixImg2Img(in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                       wavelet='db1', patch_features='l1',
                       restore_orig_size=False, min_size=(33,33)
                       ),
        # T.nn.UpsamplingNearest2d((224, 224)),
        timm.create_model(
            'efficientnetv2_m',
            in_chans=int(in_ch), num_classes=int(out_ch), pretrained=False),
    )),
    ('deepfix_efficientnet-b0', str, str, str, str, str): (lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
        DeepFixImg2Img(in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                       wavelet='db1', patch_features='l1',
                       restore_orig_size=False, min_size=(64,64)
                       ),
        get_efficientnetv1('efficientnet-b0', 'imagenet', int(in_ch), int(out_ch)))),
    ('vip_s7', str, str): (
        lambda in_ch, out_ch: T.nn.Sequential(
            T.nn.UpsamplingNearest2d((224,224)),
            vip.vip_s7(in_chans=int(in_ch), num_classes=int(out_ch))
        )),
    ('deepfix_vip_s7', str, str, str, str, str): (
        lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
            DeepFixImg2Img(in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                           wavelet='db1', patch_features='l1',
                           restore_orig_size=False, min_size=(64,64)
                           ),
            T.nn.UpsamplingNearest2d((224,224)),
            vip.vip_s7(in_chans=int(in_ch), num_classes=int(out_ch))
        )),
    ('coatnet_1_224', str, str): (
        lambda in_ch, out_ch: T.nn.Sequential(
            T.nn.UpsamplingNearest2d((224,224)),
            timm.create_model(
                'coatnet_1_224', in_chans=int(in_ch), num_classes=int(out_ch)))
    ),
    ('deepfix_coatnet_1_224', str, str, str, str, str): (
        lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
            DeepFixImg2Img(
                in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                wavelet='db1', patch_features='l1', min_size=(224,224)),
            T.nn.UpsamplingNearest2d((224,224)),
            timm.create_model(
                'coatnet_1_224', in_chans=int(in_ch),
                num_classes=int(out_ch)))),
    ('mdmlp_320', str, str): (
        lambda in_ch, out_ch: MDMLP(
            img_size=320, in_chans=int(in_ch), num_classes=int(out_ch),
            base_dim=64, depth=8, patch_size=20, overlap=10)),
    #  ('waveletres18v2', str, str, str): lambda pretrain, in_ch, out_ch: (
        #  DeepFixCompression(levels=8, wavelet='coif1', patch_size=1),
        #  R2(pretrain, int(in_ch), int(out_ch))),
    ('deepfix_mdmlp_320', str, str ,str, str, str):
        lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
            DeepFixImg2Img(
                in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                wavelet='db1', patch_features='l1', restore_orig_size=True),
            MDMLP(img_size=320, in_chans=int(in_ch), num_classes=int(out_ch),
                  base_dim=64, depth=8, patch_size=20, overlap=10)),
    ('mdmlp_patch14_lap7_dim64_depth8_224', str, str): (
        lambda in_ch, out_ch: timm.create_model(
            'mdmlp_patch14_lap7_dim64_depth8_224',
            in_chans=int(in_ch), num_classes=int(out_ch))),
    ('deepfix_mdmlp_patch14_lap7_dim64_depth8_224', str, str, str, str, str): (
        lambda in_ch, out_ch, J, Ph, Pw: T.nn.Sequential(
            DeepFixImg2Img(
                in_channels=int(in_ch), J=int(J), P=(int(Ph), int(Pw)),
                wavelet='db1', patch_features='l1', restore_orig_size=True),
            timm.create_model(
                'mdmlp_patch14_lap7_dim64_depth8_224',
                in_chans=int(in_ch), num_classes=int(out_ch)))),
    ('blur_efficientnet-b0', str, str, str): (lambda in_ch, out_ch, kernel_size: T.nn.Sequential(
        GaussianBlur(int(kernel_size) + 1 - int(kernel_size)%2, float(kernel_size)),
        T.nn.AvgPool2d(kernel_size=int(kernel_size), stride=int(kernel_size), padding=0),
        UpsamplingNearestMinSize((32,32)),
        get_efficientnetv1('efficientnet-b0', 'imagenet', int(in_ch), int(out_ch)))),

    ('medianpool2d_efficientnet-b0', str, str, str): (lambda in_ch, out_ch, kernel_size:
    T.nn.Sequential(
        MedianPool2d(kernel_size=int(kernel_size), stride=int(kernel_size), padding=0, min_size=(64,64)),
        get_efficientnetv1('efficientnet-b0', 'imagenet', int(in_ch), int(out_ch)))),
}

class UpsamplingNearestMinSize(T.nn.Module):
    """Ensure the input image has a minimum shape"""
    def __init__(self, min_size):
        super().__init__()
        self.upsample = T.nn.UpsamplingNearest2d(min_size)
        self.min_size = min_size
    def forward(self, x):
        if x.shape[-1] < self.min_size[-1] or x.shape[-2] < self.min_size[-2]:
            x = self.upsample(x)
        return x


class LossCheXpertUignore(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = T.nn.BCEWithLogitsLoss()

    def forward(self, yhat, y):
        ignore = (y != 2)  # ignore uncertainty labels
        return self.bce(yhat[ignore], y[ignore])

class KimEyeCELoss(T.nn.Module):
    def __init__(self):
        super().__init__()
        cd = T.tensor([788, 289, 467.])  # KimEye.CLASS_DISTRIBUTION
        self.ce = T.nn.CrossEntropyLoss(weight=cd.max() / cd)

    def forward(self, yhat, y):
        return self.ce(yhat, y)

def loss_intelmobileodt(yhat, y):
    """BCE Loss with class balancing weights.

    Not sure this actually helps

    because Type 2 is the hardest class, it
    has the most samples, and it separates Type 1 from Type 3.  Arguably, Type 2
    samples are on the decision boundary between Type 1 and 3.
    Class balancing weights make it harder to focus on class 2.
    """
    #  assert y.shape == yhat.shape, 'sanity check'
    #  assert y.dtype == yhat.dtype, 'sanity check'

    # class distribution of stage='train'
    w = T.tensor([249, 781, 450], dtype=y.dtype, device=y.device)
    w = (w.max() / w).reshape(1, 3)
    # w can have any of the shapes:  (B,1) or (1,C) or (B,C)
    #  return T.nn.functional.binary_cross_entropy_with_logits(yhat, y, weight=w)
    return T.nn.functional.cross_entropy(yhat, y, weight=w)
    # can't apply focal loss unless do it manually.

LOSS_FNS = {
    ('BCEWithLogitsLoss', ): lambda _: T.nn.BCEWithLogitsLoss(),
    ('CrossEntropyLoss', ): lambda _: T.nn.CrossEntropyLoss(),
    ('CE_intelmobileodt', ): lambda _: loss_intelmobileodt,
    ('chexpert_uignore', ): lambda _: LossCheXpertUignore(),
    ('kimeye_ce', ): lambda _: KimEyeCELoss(),
}

DSETS = {
    ('kimeye', str, str): get_dset_kimeye,
    ('intel_mobileodt', str, str, str, str): (
        lambda train, val, test, aug: get_dset_intel_mobileodt(train, val, test, aug)),
    #  ('origa', ... todo): ( lambda ...: get_dset_origa(...)
    #  ('riga', ... todo): ( lambda ...: get_dset_riga(...)
    ('chexpert', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=False, labels=labels)),
    ('chexpert_small', str, str): (
        lambda train_frac, val_frac: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels='diagnostic')),
    ('chexpert_small', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels=labels)),
    # chexpert_small:.1:.1:diagnostic  # all 14 classes
    # chexpert_small:.1:.1:leaderboard  # only 5 classes
    # chexpert_small:.1:.1:Cardiomegaly,Pneumonia,Pleural_Effusion  # only the defined classes
    ('chexpert_small15k', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels=labels, epoch_size=15000)),
    ('chexpert15k', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=False, labels=labels, epoch_size=15000)),
    ('flowers102', ): lambda _: get_dset_flowers102(),
    ('food101', ): (lambda _: get_dset_food101()),
    ('food101', str, str): (
        lambda J, P: get_dset_food101_deepfixed(int(J), int(P))),
}


def match(spec:str, dct:dict):
    return pampy.match(spec.split(':'), *(x for y in dct.items() for x in y))


def reset_optimizer(opt_spec:str, model:T.nn.Module) -> T.optim.Optimizer:
    spec = opt_spec.split(':')
    if opt_spec.startswith('AdaBound'):
        import adabound  # trying this out.
        kls = adabound.AdaBound
    else:
        kls = getattr(T.optim, spec[0])
    params = [(x,float(y)) for x,y in [kv.split('=') for kv in spec[1:]]]
    optimizer = kls(model.parameters(), **dict(params))
    return optimizer


def get_model_opt_loss(
        model_spec:str, opt_spec:str, loss_spec:str, regularizer_spec:str,
        device:str) -> dict[str, Union[T.nn.Module, T.optim.Optimizer]]:
    """
    Args:
        model_spec: a string of form,
            "model_name:pretraining:in_channels:out_classes".  For example:
            "effnetv2:untrained:1:5"
        opt_spec: Specifies how to create optimizer.
            First value is a pytorch Optimizer in T.optim.*.
            Other values are numerical parameters.
            Example: "SGD:lr=.003:momentum=.9"
        device: e.g. 'cpu' or 'gpu'
    Returns:
        a pytorch model and optimizer
    """
    mdl = match(model_spec, MODELS)
    mdl = mdl.to(device, non_blocking=True)
    optimizer = reset_optimizer(opt_spec, mdl)
    loss_fn = match(loss_spec, LOSS_FNS)
    if regularizer_spec != 'none':
        loss_fn = RegularizedLoss(mdl, loss_fn, regularizer_spec)
    loss_fn = loss_fn.to(device, non_blocking=True)
    return dict(model=mdl, optimizer=optimizer, loss_fn=loss_fn)


class RegularizedLoss(T.nn.Module):
    def __init__(self, model, lossfn, regularizer_spec:str):
        super().__init__()
        self.lossfn = lossfn
        self.regularizer_spec = regularizer_spec
        if regularizer_spec == 'none':
            self.regularizer = lambda *y: 0
        elif regularizer_spec.startswith('deepfixmlp:'):
            lbda = float(regularizer_spec.split(':')[1])
            self.regularizer = lambda *y: (
                float(lbda) * DeepFixMLP.get_VecAttn_regularizer(model))
        else:
            raise NotImplementedError(regularizer_spec)

    def forward(self, yhat, y):
        a, b = self.lossfn(yhat, y), self.regularizer(yhat, y)
        #  print(a.item(),b.item())
        return a + b

    def __repr__(self):
        return f'RegularizedLoss<{repr(self.lossfn)},{self.regularizer_spec}>'


def get_dset_loaders_resultfactory(dset_spec:str, device:str) -> dict:
    dct, class_names = match(dset_spec, DSETS)
    if any(dset_spec.startswith(x) for x in {'intel_mobileodt:', }):
        #  dct['result_factory'] = lambda: TL.MultiLabelBinaryClassification(
                #  class_names, binarize_fn=lambda yh: (T.sigmoid(yh)>.5).long())
        dct['result_factory'] = lambda: TL.MultiClassClassification(
            len(class_names),
            preprocess_fn=lambda yh: yh.softmax(1),
            binarize_fn=lambda yh: yh.argmax(1))
        dct['checkpoint_if'] = TL.CheckpointIf(metric='val_ROC_AUC', mode='max')
    elif any(dset_spec.startswith(x) for x in {
            'chexpert:', 'chexpert_small:',
            'chexpert_small15k:', 'chexpert15k:'}):
        dct['result_factory'] = lambda: CheXpertMultiLabelBinaryClassification(
            class_names, binarize_fn=lambda yh, th: (yh.sigmoid()>th).long(), report_avg=True, device=device)
        dct['checkpoint_if'] = TL.CheckpointIf(metric='val_ROC_AUC AVG', mode='max')
    elif dset_spec.startswith('kimeye:'):
        dct['result_factory'] = lambda: TL.MultiClassClassification(
            len(class_names),
            preprocess_fn=lambda yh: yh.softmax(1),
            binarize_fn=lambda yh: yh.argmax(1))
        dct['checkpoint_if'] = TL.CheckpointIf(metric='val_ROC_AUC', mode='max')
    elif dset_spec.startswith('flowers102'):
        dct['result_factory'] = lambda: TL.MultiClassClassification(
            len(class_names),
            preprocess_fn=lambda yh: yh.softmax(1),
            binarize_fn=lambda yh: yh.argmax(1))
        dct['checkpoint_if'] = TL.CheckpointIf(metric='val_ROC_AUC', mode='max')
    elif dset_spec.startswith('food101'):
        dct['result_factory'] = lambda: TL.MultiClassClassification(
            len(class_names),
            preprocess_fn=lambda yh: yh.softmax(1),
            binarize_fn=lambda yh: yh.argmax(1))
        dct['checkpoint_if'] = TL.CheckpointIf(metric='train_ROC_AUC', mode='max')
    else:
        raise NotImplementedError(f"I don't know how to create the result factory for {dset_spec}")
    return dct

class CheXpertMultiLabelBinaryClassification(TL.MultiLabelBinaryClassification):
    def update(self, yhat, y, loss) -> None:
        yhat, y, loss = yhat.detach(), y.detach(), loss.detach()
        loss = loss.to('cpu', non_blocking=True)
        self.num_samples += yhat.shape[0]
        assert yhat.shape == y.shape
        assert yhat.ndim == 2 and yhat.shape[1] == len(self._cms), "sanity check: model outputs expected prediction shape"
        binarized = self._binarize_fn(yhat, .5)
        assert binarized.dtype == T.long, 'sanity check binarize fn'
        assert binarized.shape == y.shape, 'sanity check binarize fn'
        ignore = (y != 2)  # ignore uncertainty labels
        for i, (kls, cm) in enumerate(self._cms.items()):
            rows = ignore[:, i]
            if rows.sum() == 0:
                continue  # don't update a confusion matrix if all data for this class is ignored
            #  print(y.device, binarized.device)
            cm += metrics.confusion_matrix_binary_soft_assignment(y[rows, i], binarized[rows, i]).to(self.device, non_blocking=True)
            if 'y' in self.metrics or 'ROC_AUC' in self.metrics:
                self._ydata[kls].append(y[rows, i].to(self.device, non_blocking=True))
            if 'yhat' in self.metrics or 'ROC_AUC' in self.metrics:
                self._yhatdata[kls].append(yhat[rows, i].to(self.device, non_blocking=True))
        self.loss += loss.item()


def train_config(args:'TrainOptions') -> TL.TrainConfig:
    dct = dict(
        **get_model_opt_loss(
            args.model, args.opt, args.lossfn, args.loss_reg, args.device),
        **get_dset_loaders_resultfactory(args.dset, args.device),
        device=args.device,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        experiment_id=args.experiment_id,
    )
    return TL.TrainConfig(**dct)


@dc.dataclass
class TrainOptions:
    """High-level configuration for training PyTorch models
    on the CheXpert or IntelMobileODTCervical datasets.
    """
    epochs:int = 50
    start_epoch:int = 0  # if "--start_epoch 1", then don't evaluate perf before training.
    device:str = 'cuda' if T.cuda.is_available() else 'cpu'

    dset:str = None
    """
      Choose the dataset.  Some options:
          --dset intel_mobileodt:train:val:test:v1
          --dset intel_mobileodt:train+additional:val:test:v1
          --dset intel_mobileodt:train+additional:noval:test:v1
          --dset chexpert:T:V:LABELS  where T + V <= 1 are the percent of training data to use for train and val, and where LABELS is one of {"diagnostic", "leaderboard"} or any comma separated list of class names (replace space with underscore, case sensitive).
          --dset chexpert_small:T:V:LABELS  the 11gb chexpert dataset.
          --dset chexpert_small:.1:.1:Cardiomegaly  # for example
          --dset chexpert_small15k:.1:.1:Cardiomegaly  # for example
    """

    opt:str = 'SGD:lr=.001:momentum=.9:nesterov=1'

    lossfn:str = None
    """
     Choose a loss function
          --lossfn BCEWithLogitsLoss
          --lossfn CrossEntropyLoss
          --lossfn CE_intelmobileodt
          --lossfn chexpert_uignore
    """

    loss_reg:str = 'none'  # Optionally add a regularizer to the loss.  loss + reg.  Accepted values:  "none", "deepfixmlp:X" where X is a positive float denoting the lambda in l1 regularizer
    model:str = 'resnet18:imagenet:3:3'  # Model specification adheres to the template "model_name:pretraining:in_ch:out_ch"
    experiment_id:str = os.environ.get('run_id', 'debugging')
    prune:str = 'off'


def main():
    p = ArgumentParser()
    p.add_arguments(TrainOptions, dest='TrainOptions')
    args = p.parse_args().TrainOptions
    print(args)
    cfg = train_config(args)

    cfg.train(cfg)

    # for x,y in cfg.train_loader:
        # plt.imshow(np.array(x.cpu())[0].transpose(1,2,0))
        # plt.pause(1)
    #  import IPython ; IPython.embed() ; import sys ; sys.exit()

    # cfg.train_dset.transform = None
    # cfg.test_dset.transform = None
    # su, sq, N = 0, 0, 0
    # mu2 = 0
    # for n,(x, _) in enumerate(cfg.train_dset):
    #     # x = x.permute(1,2,0).cpu().numpy()
    #     x = np.array(x)/255
    #     mu2 += x.mean((0,1))
    #     su += x.sum((0,1))
    #     sq += (x**2).sum((0,1))
    #     N += x.shape[0] * x.shape[1]
    #     # if n > 1000:
    #         # break
    # # print(np.array(x).shape)
    # # plt.imshow(np.array(x))
    # # plt.show()
    # mu2 = mu2/(n+1)
    # print("VAL")
    # print('std', (sq/N - (su/N)**2)**.5)
    # print('mu', su/N)
    # print('mu2', mu2)
    # # TODO: copy this to dset.py flowers102 std section..
    return cfg


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()
    cfg = main()
