#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Boilerplate to implement training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets
"""
#  import json
import os
from os.path import exists
import pampy
import sys
from simple_parsing import ArgumentParser, choice
from simplepytorch import datasets as D
from simplepytorch import trainlib as TL
from simplepytorch import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional
from sklearn.model_selection import StratifiedKFold
import dataclasses as dc
import numpy as np
import torch as T
import torchvision.transforms as tvt

from deepfix.models import get_effnetv2, get_resnet, get_densenet, get_efficientnetv1, get_DeepFixEnd2End, DeepFixMLP
from deepfix.models.ghaarconv import convert_conv2d_to_gHaarConv2d
from deepfix.init_from_distribution import init_from_beta, reset_optimizer
from deepfix import deepfix_strategies as dfs
from deepfix.models import DeepFixCompression
import pytorch_wavelets as pyw


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

import torch
import argparse
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# In[2]:


import numpy as np
import scipy.io
# from ipynb.fs.defs.q2 import *
import sklearn
import torch
import torchvision.datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import copy
import pandas as pd
from tqdm import tqdm
import time
# YOUR CODE HERE
# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print("device = {}".format(device))


# In[3]:


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
    ('waveletres18', str, str, str): lambda pretrain, in_ch, out_ch: R(
        pretrain, int(in_ch), int(out_ch)),
    ('waveletmlp', str, str, str, str, str, str, str): (
        lambda mlp_channels, in_ch, out_ch, wavelet_levels, patch_size, in_ch_mul, mlp_depth: get_DeepFixEnd2End(
            int(in_ch), int(out_ch),
            in_ch_multiplier=int(in_ch_mul), wavelet='db1',
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            mlp_depth=int(mlp_depth), mlp_channels=int(mlp_channels),
            mlp_fix_weights='none', mlp_activation=None)
        ),
    ('waveletmlpV2', str, str, str, str, str, str): (
        lambda in_ch, out_ch, wavelet, wavelet_levels, patch_size, patch_features: get_DeepFixEnd2End(
            int(in_ch), int(out_ch),
            in_ch_multiplier=1, wavelet=wavelet,
            wavelet_levels=int(wavelet_levels), wavelet_patch_size=int(patch_size),
            mlp_depth=2, mlp_channels=300,
            mlp_fix_weights='none', mlp_activation=None,
            patch_features=patch_features)
        ),

    #  ('waveletres18v2', str, str, str): lambda pretrain, in_ch, out_ch: (
        #  DeepFixCompression(levels=8, wavelet='coif1', patch_size=1),
        #  R2(pretrain, int(in_ch), int(out_ch))),
}


class R(T.nn.Module):
    def __init__(self, pretrain, in_ch, out_ch):
        super().__init__()
        self.r = get_resnet('resnet18', pretrain, in_ch, out_ch,)
        self.dwt = pyw.DWT(J=8, wave='coif1', mode='zero')

    @staticmethod
    def wavelet_coefficients_as_tensorimage(approx, detail, normalize=False):
        B,C = approx.shape[:2]
        fixed_dims = approx.shape[:-2] # num images in minibatch, num channels, etc
        output_shape = fixed_dims + (
            detail[0].shape[-2]*2,  # input img height
            detail[0].shape[-1]*2)  # input img width
        im = T.zeros(output_shape, device=approx.device, dtype=approx.dtype)
        if normalize:
            norm11 = lambda x: (x / max(x.min()*-1, x.max()))  # into [-1,+1] preserving sign
            #  approx = norm11(approx)
        im[..., :approx.shape[-2], :approx.shape[-1]] = approx if approx is not None else 0
        for level in detail:
            lh, hl, hh = level.unbind(-3)
            h,w = lh.shape[-2:]
            if normalize:
                lh, hl, hh = [norm11(x) for x in [lh, hl, hh]]
            #  im[:h, :w] = approx
            im[..., 0:h, w:w+w] = lh  # horizontal
            im[..., h:h+h, :w] = hl  # vertical
            im[..., h:h+h, w:w+w] = hh  # diagonal
        return im

    def forward(self, x):
        x = self.wavelet_coefficients_as_tensorimage(*self.dwt(x))
        return self.r(x)


class R2(T.nn.Module):
    def __init__(self, pretrain, in_ch, out_ch):
        super().__init__()
        self.r = get_resnet('resnet18', pretrain, in_ch, out_ch,)

    def forward(self, x):
        B,C,H = x.shape
        x = x.unsqueeze(-1).repeat(1,1,1,H)
        return self.r(x)


class LossCheXpertUignore(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = T.nn.BCEWithLogitsLoss()

    def forward(self, yhat, y):
        ignore = (y != 2)  # ignore uncertainty labels
        return self.bce(yhat[ignore], y[ignore])


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


def onehot(y, nclasses):
    return T.zeros((y.numel(), nclasses), dtype=y.dtype, device=y.device)            .scatter_(1, y.unsqueeze(1), 1)


def _upsample_pad_minibatch_imgs_to_same_size(batch, target_is_segmentation_mask=False):
    """a collate function for a dataloader of (x,y) samples.  """
    shapes = [item[0].shape for item in batch]
    H = max(h for c,h,w in shapes)
    W = max(w for c,h,w in shapes)
    X, Y = [], []
    for item in batch:
        h,w = item[0].shape[1:]
        dh, dw = (H-h), (W-w)
        padding = (dw//2, dw-dw//2, dh//2, dh-dh//2, )
        X.append(T.nn.functional.pad(item[0], padding))
        if target_is_segmentation_mask:
            Y.append(T.nn.functional.pad(item[1], padding))
        else:
            Y.append(item[1])
    return T.stack(X), T.stack(Y)

class RandomSampler(T.utils.data.Sampler):
    """Randomly sample without replacement a subset of N samples from a dataset
    of M>N samples.  After a while, a new subset of N samples is requested but
    (nearly) all M dataset samples have been used.  When this happens, reset
    the sampler, but ensure that N samples are returned.
    (In this case, the same image may appear in the that batch twice).

    This is useful to create smaller epochs, where each epoch represents only N
    randomly chosen images of a dataset of M>N images, and where random
    sampling is without replacement.

    """
    def __init__(self, data_source, num_samples:int=None):
        """
        Args:
            data_source:  pytorch Dataset class or object with __len__
                          ****Assume len(data_source) does not change.****
        """
        super().__init__(data_source)
        assert num_samples > 0
        assert num_samples <= len(data_source)
        self.dset_length = len(data_source)
        self.num_samples = num_samples
        self._cur_idx = 0
        self._ordering = self.new_ordering()

    def new_ordering(self):
        return T.randperm(self.dset_length, dtype=T.long)

    def next_idxs(self, _how_many=None):
        if self._cur_idx + self.num_samples >= len(self._ordering):
            some_idxs = self._ordering[self._cur_idx:self._cur_idx+self.num_samples]
            self._ordering = self.new_ordering()
            self._cur_idx = self.num_samples-len(some_idxs)
            more_idxs = self._ordering[0:self._cur_idx]
            #  print(some_idxs, more_idxs)
            idxs = T.cat([some_idxs, more_idxs])
        else:
            idxs = self._ordering[self._cur_idx:self._cur_idx+self.num_samples]
            self._cur_idx += self.num_samples
        return idxs.tolist()

    def __iter__(self):
        yield from self.next_idxs()

    def __len__(self):
        return self.num_samples


def group_random_split(
        desired_split_fracs:list[float], group:np.ndarray, rng=None):
    # in code below, imagine we have images, each belonging to a patient
    # and we want to ensure no patient is mixed across splits.
    assert all(0 <= x <= 1 for x in desired_split_fracs), 'desired_split_fracs must contain values in (0,1)'
    assert sum(desired_split_fracs) <= 1.5, 'should satisfy sum(desired_split_fracs) <= 1+eps, with some margin for error'
    # count the patients
    patients, lookup_patient_idx, counts = np.unique(
        group, return_inverse=True, return_counts=True)
    # sanity check: if any desired split is smaller than the size of a single patient,
    # there may be sampling problems where some splits are empty.
    assert min([x for x in desired_split_fracs if x != 0]) >= np.max(counts / len(group)), f'minimum allowed split fraction is >= {np.max(counts) / len(group)}'
    # randomly shuffle the patients to get an ordering over them
    if rng is None:
        rng = np.random.default_rng()
    idxs = rng.permutation(np.arange(len(patients)))
    # compute what fraction of total images we get by considering the first N
    # patients for all N.
    fracs = np.cumsum(counts[idxs]) / len(group)
    # split the data, ensuring the patients have equal chance of appearing in either set.
    img_idxs = np.arange(len(group))
    assert len(img_idxs) == len(lookup_patient_idx), 'code bug'
    splits = []
    _n_patients_so_far = 0
    _frac_so_far = 0
    for desired_frac in desired_split_fracs:
        if desired_frac == 0:
            splits.append(np.array([]))
        else:
            # define some "cut point" / threshold at which we reach the desired_frac
            n_patients = np.digitize(desired_frac+_frac_so_far, fracs, False)
            #  print(fracs, desired_frac, _frac_so_far, n_patients, n_patients - _n_patients_so_far)
            # get the indices of the samples that correspond to these patients
            splits.append(img_idxs[np.isin(lookup_patient_idx, idxs[_n_patients_so_far:n_patients])])
            # bookkeeping
            _n_patients_so_far = n_patients
            _frac_so_far = fracs[n_patients-1]
            #  _frac_so_far += desired_frac
    return splits


def get_dset_chexpert(train_frac=.8, val_frac=.2, small=False,
                      labels:str='diagnostic', num_identities=None,
                      epoch_size:int=None
                      ):
    """
    Args:
        labels:  either "diagnostic" (the 14 classes defined as
            D.CheXpert.LABELS_DIAGNOSTIC) or "identity" ("patient", "study",
            "view", "index")
        small:  whether to use CheXpert_Small dataset (previously downsampled
            images) or the fully size dataset.
        num_identities:  used only if labels='identity'.  If
            num_identities=1000, then all patients get identified as coming
            from precisely 1 of 1000 bins.
        epoch_size:  If defined, randomly sample without replacement N images each epoch.

    Returns:
        (
        {'train_dset': ..., 'val_dset': ..., 'test_dset': ...,
         'train_loader': ..., 'val_loader': ..., 'test_loader': ...
         },

        ('Pneumonia', 'Cardiomegaly', ...)  # class names defined by `labels`
        )
    """
    _label_cleanup_dct = dict(D.CheXpert.LABEL_CLEANUP_DICT)
    if labels == 'identity':
        class_names = list(range(num_identities))
        get_ylabels = lambda dct:                 (D.CheXpert.format_labels(dct, labels=['index']) % num_identities).long()
    else:
        if labels == 'diagnostic':
            class_names = D.CheXpert.LABELS_DIAGNOSTIC
        elif labels == 'leaderboard':
            class_names = D.CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD
        else:
            class_names = [x.replace('_', ' ') for x in labels.split(',')]
            assert all(x in D.CheXpert.LABELS_ALL for x in class_names), f'unrecognized class names: {labels}'
        for k in class_names:
            if k in D.CheXpert.LABELS_DIAGNOSTIC:
                _label_cleanup_dct[k][np.nan] = 0  # remap missing value to negative
        get_ylabels = lambda dct:                 D.CheXpert.format_labels(dct, labels=class_names).float()
    kws = dict(
        img_transform=tvt.Compose([
            tvt.ToTensor(),  # full res 1024x1024 imgs
#           tvt.CenterCrop((160,160)),
            tvt.Resize((160, 160)),
            
#             tvt.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]),
        getitem_transform=lambda dct: (dct['image'], get_ylabels(dct)),
        label_cleanup_dct=_label_cleanup_dct,
    )
    if small:
        kls = D.CheXpert_Small
    else:
        kls = D.CheXpert

    train_dset = kls(dataset_dir="../data/CheXpert-v1.0-small/",use_train_set=True, **kws)
    N = len(train_dset)
    # split the dataset into train and val sets
    # ensure patient images exist only in one set.  no mixing.
    train_idxs, val_idxs = group_random_split(
        [train_frac, val_frac], group=train_dset.labels_csv['Patient'].values)
    val_dset = T.utils.data.Subset(train_dset, val_idxs)
    train_dset = T.utils.data.Subset(train_dset, train_idxs)
    # the 200 sample "test" set
    test_dset = kls(dataset_dir="../data/CheXpert-v1.0-small/",use_train_set=False, **kws)
    # data loaders
    batch_size = int(os.environ.get('batch_size', 1600))
    batch_dct = dict(
#         collate_fn=_upsample_pad_minibatch_imgs_to_same_size,
        num_workers=int(os.environ.get("num_workers", 5)))  # upsample pad must take time
    train_loader=DataLoader(
        train_dset, batch_size=batch_size,
        sampler=RandomSampler(train_dset, epoch_size or len(train_dset)), **batch_dct)
    val_loader=DataLoader(val_dset, batch_size=batch_size, **batch_dct)
    test_loader=DataLoader(test_dset, batch_size=1, **batch_dct)
    #
    # debugging:  vis dataset
    #  from deepfix.plotting import plot_img_grid
    #  from matplotlib import pyplot as plt
    #  plt.ion()
    #  fig, ax = plt.subplots(1,2)
    #  print('hello world')
    #  for mb in train_loader:
        #  plot_img_grid(mb[0].squeeze(1), num=1, suptitle=f'shape: {mb[0].shape}')
        #  plt.show(block=False)
        #  plt.pause(1)
    #
    return (dict(
        train_dset=train_dset, val_dset=val_dset, test_dset=test_dset,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    ), class_names)




LOSS_FNS = {
    ('BCEWithLogitsLoss', ): lambda _: T.nn.BCEWithLogitsLoss(),
    ('CrossEntropyLoss', ): lambda _: T.nn.CrossEntropyLoss(),
    ('CE_intelmobileodt', ): lambda _: loss_intelmobileodt,
    ('chexpert_uignore', ): lambda _: LossCheXpertUignore(),
}

DSETS = {
    ('intel_mobileodt', str, str, str, str): (
        lambda train, val, test, aug: get_dset_intel_mobileodt(train, val, test, aug)),
    #  ('origa', ... todo): ( lambda ...: get_dset_origa(...)
    #  ('riga', ... todo): ( lambda ...: get_dset_riga(...)
    ('chexpert', str, str): (
        lambda train_frac, val_frac: get_dset_chexpert(
            float(train_frac), float(val_frac), small=False, labels='diagnostic')),
    ('chexpert_small', str, str): (
        lambda train_frac, val_frac: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels='diagnostic')),
    ('chexpert_small', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels=labels)),
    # chexpert_small:.1:.1:diagnostic  # all 14 classes
    # chexpert_small:.1:.1:leaderboard  # only 5 classes
    # chexpert_small:.1:.1:Cardiomegaly,Pneumonia,Pleural_Effusion  # only the defined classes
    #  'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', ...
    ('chexpert_small15k', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels=labels, epoch_size=15000)),
}


def match(spec:str, dct:dict):
    return pampy.match(spec.split(':'), *(x for y in dct.items() for x in y))


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
        return self.lossfn(yhat, y) + self.regularizer(yhat, y)

    def __repr__(self):
        return f'RegularizedLoss<{repr(self.lossfn)},{self.regularizer_spec}>'


def get_dset_loaders_resultfactory(dset_spec:str) -> dict:
    dct, class_names = match(dset_spec, DSETS)
    if any(dset_spec.startswith(x) for x in {'intel_mobileodt:', }):
        #  dct['result_factory'] = lambda: TL.MultiLabelBinaryClassification(
                #  class_names, binarize_fn=lambda yh: (T.sigmoid(yh)>.5).long())
        dct['result_factory'] = lambda: TL.MultiClassClassification(
                len(class_names), binarize_fn=lambda yh: yh.softmax(1).argmax(1))
    elif any(dset_spec.startswith(x) for x in {
            'chexpert:', 'chexpert_small:', 'chexpert_small15k:'}):
        dct['result_factory'] = lambda: CheXpertMultiLabelBinaryClassification(
            class_names, binarize_fn=lambda yh: (yh.sigmoid()>.5).long(), report_avg=True)
    else:
        raise NotImplementedError(f"I don't know how to create the result factory for {dset_spec}")
    return dct

class CheXpertMultiLabelBinaryClassification(TL.MultiLabelBinaryClassification):
    def update(self, yhat, y, loss) -> None:
        self.num_samples += yhat.shape[0]
        self.loss += loss.item()
        assert yhat.shape == y.shape
        assert yhat.ndim == 2 and yhat.shape[1] == len(self._cms), "sanity check: model outputs expected prediction shape"
        binarized = self._binarize_fn(yhat)
        assert binarized.dtype == T.long, 'sanity check binarize fn'
        assert binarized.shape == y.shape, 'sanity check binarize fn'
        ignore = (y != 2)  # ignore uncertainty labels
        for i, (kls, cm) in enumerate(self._cms.items()):
            rows = ignore[:, i]
            if rows.sum() == 0:
                continue  # don't update a confusion matrix if all data for this class is ignored
            cm += metrics.confusion_matrix(y[rows, i], binarized[rows, i], num_classes=2).cpu()


def get_deepfix_train_strategy(args:'TrainOptions'):
    deepfix_spec = args.deepfix
    if deepfix_spec == 'off':
        return TL.train_one_epoch
    elif deepfix_spec.startswith('reinit:'):
        _, N, P, R = deepfix_spec.split(':')
        return dfs.DeepFix_TrainOneEpoch(int(N), float(P), int(R), TL.train_one_epoch)
    elif deepfix_spec.startswith('dhist:'):
        fp = deepfix_spec.split(':', 1)[1]
        assert exists(fp), f'histogram file not found: {fp}'
        return dfs.DeepFix_DHist(fp)
    elif deepfix_spec.startswith('dfhist:'):
        fp = deepfix_spec.split(':', 1)[1]
        assert exists(fp), f'histogram file not found: {fp}'
        return dfs.DeepFix_DHist(fp, fixed=True)
    elif deepfix_spec == 'fixed':
        return dfs.DeepFix_DHist('', fixed=True, init_with_hist=False)
    elif deepfix_spec.startswith('beta:'):
        alpha, beta = deepfix_spec.split(':')[1:]
        return dfs.DeepFix_LambdaInit(
            lambda cfg: init_from_beta(cfg.model, float(alpha), float(beta)))
    elif deepfix_spec.startswith('ghaarconv2d:'):
        ignore_layers = deepfix_spec.split(':')[1].split(',')
        return dfs.DeepFix_LambdaInit(
            lambda cfg: (
                print(f'initialize {deepfix_spec}'),
                convert_conv2d_to_gHaarConv2d(cfg.model, ignore_layers=ignore_layers),
                reset_optimizer(args.opt, cfg.model),
                print(cfg.model)
            ))
    else:
        raise NotImplementedError(deepfix_spec)


def train_config(args:'TrainOptions') -> TL.TrainConfig:
    return TL.TrainConfig(
        **get_model_opt_loss(
            args.model, args.opt, args.lossfn, args.loss_reg, args.device),
        **get_dset_loaders_resultfactory(args.dset),
        device=args.device,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        train_one_epoch=get_deepfix_train_strategy(args),
        experiment_id=args.experiment_id,
    )


def main():
    
    d,l = get_dset_chexpert()
    train_loader = d['train_loader']
    test_loader = d['test_loader']
    
    
    # DEEPFIX COMPRESSION ======================================================================================================>
    enc = DeepFixCompression(in_ch=1, in_ch_multiplier=1, levels=5,
                             wavelet='coif2',
                             patch_size=5, patch_features=('l1', ))

    X_train = []
    Y_train = []

    j, i  = 0, 0
    for data in tqdm(trainset_loader):
        j += 1
        if j == 63000:
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            pd.DataFrame(X_train).to_csv("../deepfix/feature_csv_console/wavelet_x_train_f_{}.csv".format(i), index=False) 
            pd.DataFrame(Y_train).to_csv("../deepfix/feature_csv_console/wavelet_y_train_f_{}.csv".format(i), index=False)
            X_train = []
            Y_train = []
            i += 1
            j = 0

        inputs, labels = data
        img = enc(inputs)
        X_train.append(img.numpy().reshape(-1, ))
        Y_train.append(labels.numpy().reshape(-1, ))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    pd.DataFrame(X_train).to_csv("../deepfix/feature_csv_console/wavelet_x_train_f_{}.csv".format(2), index=False)  
    pd.DataFrame(Y_train).to_csv("../deepfix/feature_csv_console/wavelet_y_train_f_{}.csv".format(2), index=False)

    
    X_test = []
    Y_test = []
    for data in tqdm(valset_loader):
        inputs, labels = data
        inputs = inputs#.to(device)
        img = enc(inputs)
        X_test.append(img.numpy().reshape(-1,))
        Y_test.append(labels.numpy().reshape(-1,))
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    pd.DataFrame(X_test).to_csv("../deepfix/feature_csv/wavelet_x_test_f.csv", index=False)  
    pd.DataFrame(Y_test).to_csv("../deepfix/feature_csv/wavelet_y_test_f.csv", index=False)


    # Read the DeepFix Compressed data =====================================================================================>     
    Y_train0 = pd.read_csv("../deepfix/feature_csv/wavelet_y_train_f_0.csv")
    Y_train0 = Y_train0.iloc[:,:].values

    Y_train1 = pd.read_csv("../deepfix/feature_csv/wavelet_y_train_f_1.csv")
    Y_train1 = Y_train1.iloc[:,:].values

    Y_train2 = pd.read_csv("../deepfix/feature_csv/wavelet_y_train_f_2.csv")
    Y_train2 = Y_train2.iloc[:,:].values

    Y_train = np.vstack((Y_train0,Y_train1,Y_train2))

    Y_test = pd.read_csv("../deepfix/feature_csv/wavelet_y_test_f.csv")
    Y_test = Y_test.iloc[:,:].values    
    print('Y_train and Y_test is ready')
    
    start = time.time()
    print('Reading x_train data')
    chunk = pd.read_csv("../deepfix/feature_csv/wavelet_x_train_f_0.csv",chunksize=5000)
    X_train0 = pd.concat(chunk)
    chunk = pd.read_csv("../deepfix/feature_csv/wavelet_x_train_f_1.csv",chunksize=5000)
    X_train1 = pd.concat(chunk)
    chunk = pd.read_csv("../deepfix/feature_csv/wavelet_x_train_f_2.csv",chunksize=5000)
    X_train2 = pd.concat(chunk)

    X_train0 = X_train0.iloc[:,:].values
    X_train1 = X_train1.iloc[:,:].values
    X_train2 = X_train2.iloc[:,:].values
    X_train = np.vstack((X_train0,X_train1,X_train2))
    print('X_train data ready, time elapsed {} min'.format((time.time()-start)/60))
    
    start = time.time()
    print('Reading X_test')
    X_test = pd.read_csv("../deepfix/feature_csv/wavelet_x_test_f.csv")
    X_test = X_test.iloc[:,:].values

    print('X_test data ready, time elapsed {} min'.format((time.time()-start)/60))
# Dataset Ready, create a dataloader ================================================================================================>    
    transform=tvt.Compose([tvt.ToTensor(),])
    
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(Y_train, dtype=torch.float32)
    
    train_data = TensorDataset(inputs, targets)#torch.utils.data
    
    inputs = torch.tensor(X_test, dtype=torch.float32)
    targets = torch.tensor(Y_test, dtype=torch.float32)
    
    test_data = TensorDataset(inputs, targets)  
    
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    print('DataLoader Created')

if __name__ == "__main__":
    main()
