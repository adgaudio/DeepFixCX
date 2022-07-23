"""
Training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets.
"""
#  import json
import os
import pampy
from simple_parsing import ArgumentParser
from simplepytorch import datasets as D
from simplepytorch import trainlib as TL
from simplepytorch import metrics
from torch.utils.data import DataLoader
from typing import Union, Tuple
import dataclasses as dc
import numpy as np
import torch as T
import torchvision.transforms as tvt

from heartspot.models import get_effnetv2, get_resnet, get_densenet, get_efficientnetv1
from heartspot.models.qthline import QTLineClassifier, HLine, RLine, MedianPoolDenseNet
from heartspot.models.quadtree import QT
from heartspot.models.median_pooling import MedianPool2d


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
    #  ('hline', ):
        #  lambda _: QTLineClassifier(HLine(list(range(100,300,5)), 320), None),
    ('hline_10', ):
        lambda _: QTLineClassifier(HLine(list(range(100,300,10)), 320), None),
    #  ('rline', ):
        #  lambda _: QTLineClassifier(RLine((320,320), nlines=200, seed=1), None),
    ('rline1', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=77, seed=1), None),
    ('rline2', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=77, seed=2), None),
    ('rline3', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=77, seed=3), None),
    ('rline3f', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=77, zero_top_frac=1/3, seed=3), None),
    ('rline2_200', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=1/3, seed=1), None),
    ('rline_200heart', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=1/3, seed=1, heart_roi=True), None),
    ('rline_200heart2', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True), None),
    ('rhline_opt', ):
        lambda _: QTLineClassifier(RLine((320,320), nlines=150, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(80,295,6))), None),
    ('qrhline', ):
        lambda _: QTLineClassifier(
            RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))),
            QT(30, 9, split_mode='entropy')
            ),
    ('qrhline_fast', ):
        lambda _: QTLineClassifier(
            RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))),
            QT(200, 9, split_mode='entropy')
            ),
    ('median', ): lambda _: MedianPoolDenseNet(),
    ('sum', ): lambda _: QTLineClassifier(
        RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=False, hlines=[], sum_aggregate=True), None),
    ('heart+sum', ): lambda _: QTLineClassifier(
        RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=[], sum_aggregate=True), None),
    ('rhline+heart+sum', ): lambda _: QTLineClassifier(
        RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10)), sum_aggregate=True), None),
    ('rhline+sum', ): lambda _: QTLineClassifier(
        RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100,300,10)), sum_aggregate=True), None),
    ('rhline+densenet', ): lambda _: T.nn.Sequential(
        RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100,300,10)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)
    ),
    # MLP Models
    ('hline', ): lambda _: QTLineClassifier(RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100,300,10))), None),
    ('rline', ): lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=False, hlines=[]), None),
    ('heart', ): lambda _: QTLineClassifier(RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=[]), None),
    ('hline+heart', ): lambda _: QTLineClassifier(RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))), None),
    ('rline+heart', ): lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True), None),
    ('rhline', ): lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100,300,10))), None),
    ('rhline+heart', ): lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))), None),
    ('rline+heart_s2', ): lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=2, heart_roi=True), None),
    ('rline+heart_s3', ): lambda _: QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=3, heart_roi=True), None),
    ('median+rhline+heart', ): lambda _: QTLineClassifier(
        RLine((320//2,320//2), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100//2,300//2,10//2))),
        quadtree=MedianPool2d(kernel_size=12, stride=2, same=True)),
    # DenseNet models
    ('hline+densenet', ): lambda _: T.nn.Sequential(
        RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100,300,10)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    ('rline+densenet', ): lambda _: T.nn.Sequential(
        RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=False, hlines=[], ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    ('heart+densenet', ): lambda _: T.nn.Sequential(
        RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=[], ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    ('rhline+heart+densenet', ): lambda _: T.nn.Sequential(
        RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    ('median+rhline+heart+densenet', ): lambda _: T.nn.Sequential(
        MedianPool2d(kernel_size=12, stride=2, same=True),
        RLine((320//2,320//2), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100//2,300//2,10//2)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    # extra models
    ('median+hline+densenet', ): lambda _: T.nn.Sequential(
        MedianPool2d(kernel_size=12, stride=2, same=True),
        RLine((320//2,320//2), nlines=0, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100//2,300//2,10//2)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    ('median+hline+heart+densenet', ): lambda _: T.nn.Sequential(
        MedianPool2d(kernel_size=12, stride=2, same=True),
        RLine((320//2,320//2), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100//2,300//2,10//2)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
    ('medians1+rhline+heart', ): lambda _: QTLineClassifier(
        RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))),
        quadtree=MedianPool2d(kernel_size=12, stride=1, same=True)),
    ('median+densenet', ): lambda _: T.nn.Sequential(
        MedianPool2d(kernel_size=12, stride=2, same=True), get_densenet('densenet121', 'untrained', 1, 1)),
    ('hline+heart+densenet', ): lambda _: T.nn.Sequential(
        RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10)), ret_img=True),
        get_densenet('densenet121', 'untrained', 1, 1)),
}


class LossCheXpertUignore(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = T.nn.BCEWithLogitsLoss()

    def forward(self, yhat, y):
        ignore = (y != 2)  # ignore uncertainty labels
        return self.bce(yhat[ignore], y[ignore])


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


class ResizeCenterCropTo(T.nn.Module):
    """Resize tensor image to desired shape yx=(y, x) by enlarging the image
    without changing its aspect ratio, then center cropping.

    This can cut out the boundaries of images.
    """
    def __init__(self, yx:Tuple[int]):
        super().__init__()
        self.yx = yx
        self.center_crop = tvt.CenterCrop(yx)

    def resize_preserve_aspect(self, x:T.Tensor):
        """
        Args:
            x: Tensor image of shape (?, H, W) or other shape accepted by
                tvt.functional.resize"""
        given_shape = x.shape[-2:]
        sufficiently_large = np.array(given_shape) >= self.yx
        if np.all(sufficiently_large):
            return x
        desired_aspect = self.yx[0] / self.yx[1]
        given_aspect = given_shape[0] / given_shape[1]
        if given_aspect < desired_aspect:
            # resize the y dim based on x
            resize_to = (self.yx[1], round(self.yx[0] / given_shape[0] * given_shape[1]))
        elif given_aspect > desired_aspect:
            # resize the x dim based on y
            resize_to = (round(self.yx[1] / given_shape[1] * given_shape[0]), self.yx[0])
        else:
            # same aspect ratio. just resize the image to yx
            resize_to = self.yx
        x = tvt.functional.resize(x, resize_to)
        #  print(x.shape)
        return x

    def forward(self, x):
        x = self.resize_preserve_aspect(x)
        x = self.center_crop(x)
        return x


def get_dset_chexpert(train_frac=.8, val_frac=.2, small=False,
                      labels:str='diagnostic', num_identities=None,
                      epoch_size:int=None
                      ):
    """
    Args:
        labels:  either "diagnostic" (the 14 classes defined as
            D.CheXpert.LABELS_DIAGNOSTIC) or "identity" ("patient", "study",
            "view", "index").
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
        get_ylabels = lambda dct: \
            (D.CheXpert.format_labels(dct, labels=['index'], as_tensor=True) % num_identities).long()
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
        get_ylabels = lambda dct: \
            D.CheXpert.format_labels(dct, labels=class_names, as_tensor=True).float()
    kws = dict(
        getitem_transform=lambda dct: (dct['image'], get_ylabels(dct)),
        label_cleanup_dct=_label_cleanup_dct,
        img_loader='cv2',
    )
    if small:
        kls = D.CheXpert_Small
        kws['img_transform'] = tvt.Compose([
            tvt.ToTensor(),
            tvt.CenterCrop((320,320)) if small else (lambda x: x),
        ])
    else:
        kls = D.CheXpert
        kws['img_transform'] = tvt.Compose([
            tvt.ToTensor(),
            #  lambda x: x.to('cuda', non_blocking=True),  # assume num_workers=0
            ResizeCenterCropTo((2320, 2320))  # preserving aspect ratio and center crop
        ])

    train_dset = kls(use_train_set=True, **kws)
    # split the dataset into train and val sets
    # ensure patient images exist only in one set.  no mixing.
    train_idxs, val_idxs = group_random_split(
        [train_frac, val_frac], group=train_dset.labels_csv['Patient'].values)
    val_dset = T.utils.data.Subset(train_dset, val_idxs)
    train_dset = T.utils.data.Subset(train_dset, train_idxs)
    # the 200 sample "test" set
    test_dset = kls(use_train_set=False, **kws)
    # data loaders
    batch_size = int(os.environ.get('batch_size', 15))
    print('batch size', batch_size)
    batch_dct = dict(
        #  pin_memory=True,
        num_workers=int(os.environ.get("num_workers", 0)))  # upsample pad must take time
    print('num workers', batch_dct['num_workers'])
    train_loader=DataLoader(
        train_dset, batch_size=batch_size,
        sampler=RandomSampler(train_dset, epoch_size or len(train_dset)), **batch_dct)
    val_loader=DataLoader(val_dset, batch_size=batch_size, **batch_dct)
    test_loader=DataLoader(test_dset, batch_size=batch_size, **batch_dct)
    #
    # debugging:  vis dataset
    #  from heartspot.plotting import plot_img_grid
    #  from matplotlib import pyplot as plt
    #  plt.ion()
    #  fig, ax = plt.subplots(1,2)
    #  print('hello world')
    #  for mb in train_loader:
    #      plot_img_grid(mb[0].squeeze(1), num=1, suptitle=f'shape: {mb[0].shape}')
    #      plt.show(block=False)
    #      plt.pause(1)
    #  #
    #  import sys ; sys.exit()
    return (dict(
        train_dset=train_dset, val_dset=val_dset, test_dset=test_dset,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    ), class_names)


LOSS_FNS = {
    ('BCEWithLogitsLoss', ): lambda _: T.nn.BCEWithLogitsLoss(),
    ('CrossEntropyLoss', ): lambda _: T.nn.CrossEntropyLoss(),
    ('chexpert_uignore', ): lambda _: LossCheXpertUignore(),
}

DSETS = {
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
    #  'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', ...
    ('chexpert_small15k', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True, labels=labels, epoch_size=15000)),
    ('chexpert15k', str, str, str): (
        lambda train_frac, val_frac, labels: get_dset_chexpert(
            float(train_frac), float(val_frac), small=False, labels=labels, epoch_size=15000)),
}


def match(spec:str, dct:dict):
    return pampy.match(spec.split(':'), *(x for y in dct.items() for x in y))


def get_model_opt_loss(
        model_spec:str, opt_spec:str, loss_spec:str,
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
    return dict(model=mdl, optimizer=optimizer, loss_fn=loss_fn)


def get_dset_loaders_resultfactory(dset_spec:str, device:str) -> dict:
    dct, class_names = match(dset_spec, DSETS)
    if any(dset_spec.startswith(x) for x in {
            'chexpert:', 'chexpert_small:',
            'chexpert_small15k:', 'chexpert15k:'}):
        dct['result_factory'] = lambda: CheXpertMultiLabelBinaryClassification(
            class_names, binarize_fn=lambda yh, th: (yh.sigmoid()>th).long(), report_avg=True, device=device)
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
    return TL.TrainConfig(
        **get_model_opt_loss(
            args.model, args.opt, args.lossfn, args.device),
        **get_dset_loaders_resultfactory(args.dset, args.device),
        device=args.device,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        experiment_id=args.experiment_id,
        checkpoint_if=TL.CheckpointIf(metric='val_ROC_AUC AVG', mode='max')
    )


@dc.dataclass
class TrainOptions:
    """High-level configuration for training PyTorch models on the CheXpert dataset
    """
    epochs:int = 50
    start_epoch:int = 0  # if "--start_epoch 1", then don't evaluate perf before training.
    device:str = 'cuda' if T.cuda.is_available() else 'cpu'

    dset:str = None
    """
      Choose the dataset.  Some options:
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
          --lossfn chexpert_uignore
    """

    model:str = 'resnet18:imagenet:3:3'  # Model specification adheres to the template "model_name:pretraining:in_ch:out_ch"
    experiment_id:str = os.environ.get('run_id', 'debugging')


def main():
    p = ArgumentParser()
    p.add_arguments(TrainOptions, dest='TrainOptions')
    args = p.parse_args().TrainOptions
    print(args)
    cfg = train_config(args)

    cfg.train(cfg)
    #  import IPython ; IPython.embed() ; import sys ; sys.exit()

    #  with T.profiler.profile(
    #      activities=[
    #          T.profiler.ProfilerActivity.CPU,
    #          T.profiler.ProfilerActivity.CUDA,
    #      ], with_modules=True,
    #  ) as p:
    #      cfg.train(cfg)
    #  print(p.key_averages().table(
    #      sort_by="self_cuda_time_total", row_limit=-1))
    return cfg


if __name__ == "__main__":
    cfg = main()
