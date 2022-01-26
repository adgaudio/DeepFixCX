"""
Boilerplate to implement training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets
"""
#  import json
import os
from os.path import exists
import pampy
from simple_parsing import ArgumentParser, choice
from simplepytorch import datasets as D
from simplepytorch import trainlib as TL
from simplepytorch import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional
import dataclasses as dc
import numpy as np
import torch as T
import torchvision.transforms as tvt

from deepfix.models import get_effnetv2, get_resnet, get_efficientnetv1, get_DeepFixEnd2End, DeepFixMLP
from deepfix.models.ghaarconv import convert_conv2d_to_gHaarConv2d
from deepfix.init_from_distribution import init_from_beta, reset_optimizer
from deepfix import deepfix_strategies as dfs
import pytorch_wavelets as pyw


MODELS = {
    ('effnetv2', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_effnetv2(pretrain, int(in_ch), int(out_ch))),
    ('resnet50', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_resnet('resnet50', pretrain, int(in_ch), int(out_ch))),
    ('resnet18', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_resnet('resnet18', pretrain, int(in_ch), int(out_ch))),
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


class LossCheXpertIdentity(T.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.bce = T.nn.BCEWithLogitsLoss()
        self.N = N

    def forward(self, yhat, y):
        # absolute max possible num patients in chexpert is 223414
        # but let's just hash them into a smaller number of bins via modulo N
        assert self.N == yhat.shape[1], \
                f'note: model must have {self.N} binary predictions per sample'
        y_onehot = y.new_zeros(y.shape[0], self.N, dtype=T.float
                               ).scatter_(1, y.long()%self.N, 1)
        return self.bce(yhat[:, -1], y_onehot[:, -1])


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
    return T.zeros((y.numel(), nclasses), dtype=y.dtype, device=y.device)\
            .scatter_(1, y.unsqueeze(1), 1)


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


def get_dset_chexpert(train_frac=.8, val_frac=.2, small=False,
                      labels:str='diagnostic', num_identities=None):
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
                (D.CheXpert.format_labels(dct, labels=['index']) % num_identities).long()
    else:
        if labels == 'diagnostic':
            class_names = D.CheXpert.LABELS_DIAGNOSTIC
        elif labels == 'leaderboard':
            class_names = D.CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD
        else:
            class_names = [x.replace('_', ' ') for x in labels.split(',')]
            assert all(x in D.CheXpert.LABELS_ALL for x in class_names), f'unrecognized class names: {labels}'
        for k in class_names:
            _label_cleanup_dct[k][np.nan] = 0  # remap missing value to negative
        get_ylabels = lambda dct: \
                D.CheXpert.format_labels(dct, labels=class_names).float()
    kws = dict(
        img_transform=tvt.Compose([
            #  tvt.RandomCrop((512, 512)),
            tvt.ToTensor(),  # full res 1024x1024 imgs
        ]),
        getitem_transform=lambda dct: (dct['image'], get_ylabels(dct)),
        label_cleanup_dct=_label_cleanup_dct,
    )
    if small:
        kls = D.CheXpert_Small
    else:
        kls = D.CheXpert

    train_dset = kls(use_train_set=True, **kws)
    N = len(train_dset)
    if train_frac + val_frac == 1:
        nsplits = [N - int(N*val_frac), int(N*val_frac), 0]
    else:
        a,b = int(N*train_frac), int(N*val_frac)
        nsplits = [a,b, N-a-b]
    train_dset, val_dset, _ = T.utils.data.random_split(train_dset, nsplits)
    test_dset = kls(use_train_set=False, **kws)
    batch_dct = dict(
        batch_size=15, collate_fn=_upsample_pad_minibatch_imgs_to_same_size,
        num_workers=int(os.environ.get("num_workers", 4)))  # upsample pad must take time
    train_loader=DataLoader(train_dset, shuffle=True, **batch_dct)
    val_loader=DataLoader(val_dset, **batch_dct)
    test_loader=DataLoader(test_dset, **batch_dct)
    return (dict(
        train_dset=train_dset, val_dset=val_dset, test_dset=test_dset,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    ), class_names)


def get_dset_intel_mobileodt(stage_trainval:str, use_val:str, stage_test:str, augment:str
                             ) -> (dict[str,Optional[Union[Dataset,DataLoader]]], tuple[str]):
    """Obtain train/val/test splits for the IntelMobileODT Cervical Cancer
    Colposcopy dataset, and the data loaders.

    Args:
        stage_trainval: the `stage` for training and validation.
            i.e. Possible choices:  {'train', 'train+additional'}
            Train / val split is 70/30 random stratified split.
        use_val: Whether to create a validation set
            Choices:  {"val", "noval"}
        stage_test: the `stage` for test set.  Should be "test".
        augment: Type of augmentations to apply.  One of {'v1', }.
            "v1" - make the aspect ratio .75, resize images to (200,150), and convert in range [0,1]
    Returns:
        (
        {'train_dset': ..., 'val_dset': ..., 'test_dset': ...,
         'train_loader': ..., 'val_loader': ..., 'test_loader': ...
         },

        ('Type 1', 'Type 2', 'Type 3')
        )
    """
    assert augment == 'v1', 'code bug: other augmentations not implemented'
    base_dir = './data/intel_mobileodt_cervical_resized'
    dset_trainval = D.IntelMobileODTCervical(stage_trainval, base_dir)
    _y = [dset_trainval.getitem(i, load_img=False)
          for i in range(len(dset_trainval))]
    dct = {'test_dset': D.IntelMobileODTCervical(stage_test, base_dir)}
    if use_val == 'noval':
        dct['train_dset'] = dset_trainval
        dct['val_dset'] = None
    else:
        assert use_val == 'val', f'unrecognized option: {use_val}'
        idxs_train, idxs_val = list(
            StratifiedShuffleSplit(1, test_size=.3).split(
                np.arange(len(dset_trainval)), _y))[0]
        dct['train_dset'] = T.utils.data.Subset(dset_trainval, idxs_train)
        dct['val_dset'] = T.utils.data.Subset(dset_trainval, idxs_val)

    # preprocess train/val/test images all the same way
    preprocess_v1 = tvt.Compose([
        #
        ### previously done (to save computation time) ###
        #  D.IntelMobileODTCervical.fix_aspect_ratio,
        #  tvt.Resize((200, 150)),  # interpolation=tvt.InterpolationMode.NEAREST),
        #
        lambda x: x.float()/255.
    ])
    dct = {k: D.PreProcess(v, lambda xy: (
        preprocess_v1(xy[0]),
        #  onehot(xy[1].unsqueeze(0).long()-1, 3).squeeze_().float()))
        xy[1].long()-1))
        for k,v in dct.items()}
    dct.update(dict(
        train_loader=DataLoader(dct['train_dset'], batch_size=20, shuffle=True),
        test_loader=DataLoader(dct['test_dset'], batch_size=20),))
    if dct['val_dset'] is None:
        dct['val_loader'] = None
    else:
        dct['val_loader'] = DataLoader(dct['val_dset'], batch_size=20)
    class_names = [x.replace('_', ' ') for x in D.IntelMobileODTCervical.LABEL_NAMES]
    return dct, class_names


LOSS_FNS = {
    ('BCEWithLogitsLoss', ): lambda _: T.nn.BCEWithLogitsLoss(),
    ('CrossEntropyLoss', ): lambda _: T.nn.CrossEntropyLoss(),
    ('CE_intelmobileodt', ): lambda _: loss_intelmobileodt,
    ('chexpert_uignore', ): lambda _: LossCheXpertUignore(),
    ('chexpert_identity', str): lambda out_ch: LossCheXpertIdentity(N=int(out_ch)),
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

    ('chexpert_small_ID', str, str, str): (
        lambda num_identities, train_frac, val_frac: get_dset_chexpert(
            float(train_frac), float(val_frac), small=True,
            labels='identity', num_identities=int(num_identities))),
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
    if any(dset_spec.startswith(x) for x in {'intel_mobileodt:',
                                             'chexpert_small_ID:'}):
        #  dct['result_factory'] = lambda: TL.MultiLabelBinaryClassification(
                #  class_names, binarize_fn=lambda yh: (T.sigmoid(yh)>.5).long())
        dct['result_factory'] = lambda: TL.MultiClassClassification(
                len(class_names), binarize_fn=lambda yh: yh.softmax(1).argmax(1))
    elif any(dset_spec.startswith(x) for x in {'chexpert:', 'chexpert_small:'}):
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


@dc.dataclass
class TrainOptions:
    """High-level configuration for training PyTorch models
    on the IntelMobileODTCervical dataset.
    """
    epochs:int = 50
    start_epoch:int = 0  # if "--start_epoch 1", then don't evaluate perf before training.
    device:str = 'cuda' if T.cuda.is_available() else 'cpu'
    dset:str = None #choice(
        #  'intel_mobileodt:train:val:test:v1',
        #  'intel_mobileodt:train+additional:val:test:v1',
        #  'intel_mobileodt:train+additional:noval:test:v1',
        #  'chexpert:.8:.2', 'chexpert:.01:.01', 'chexpert:.001:.001',
        #  'chexpert_small:.8:.2', 'chexpert_small:.01:.01',
        #   'chexpert_small:.001:.001',
        #  default='intel_mobileodt:train:val:test:v1')
    opt:str = 'SGD:lr=.001:momentum=.9:nesterov=1'
    lossfn:str = None  # choices:
        #  'BCEWithLogitsLoss',
        #  'CrossEntropyLoss', 
        #  'CE_intelmobileodt',
        #  'chexpert_uignore', 
        #  'chexpert_identity:N' for some N=num_identities predicted by model (compared to identities y%N)
    loss_reg:str = 'none'  # Optionally add a regularizer to the loss.  loss + reg.  Accepted values:  'none', 'deepfixmlp:X' where X is a positive float denoting the lambda in l1 regularizer
    model:str = 'resnet18:imagenet:3:3'  # Model specification adheres to the template "model_name:pretraining:in_ch:out_ch"
    deepfix:str = 'off'  # DeepFix Re-initialization Method.
                         #  "off" or "reinit:N:P:R" or "d[f]hist:path_to_histogram.pth"
                         #  or "beta:A:B" for A,B as (float) parameters of the beta distribution
                         # 'ghaarconv2d:layer1,layer2' Replaces all spatial convolutions with GHaarConv2d layer except the specified layers
    experiment_id:str = os.environ.get('run_id', 'debugging')
    prune:str = 'off'

    def execute(self):
        cfg = train_config(self)
        cfg.train(cfg)


def main():
    p = ArgumentParser()
    p.add_arguments(TrainOptions, dest='TrainOptions')
    args = p.parse_args().TrainOptions
    print(args)
    cfg = train_config(args)

    if args.prune != 'off':
        assert args.prune.startswith('ChannelPrune:')
        raise NotImplementedError('code is a bit hardcoded, so it is not available without hacking on it.')
        print(args.prune)
        from explainfix import channelprune
        from deepfix.weight_saliency import costfn_multiclass
        a = sum([x.numel() for x in cfg.model.parameters()])
        channelprune(cfg.model, pct=5, grad_cost_fn=costfn_multiclass,
                     loader=cfg.train_loader, device=cfg.device, num_minibatches=10)
        b = sum([x.numel() for x in cfg.model.parameters()])
        assert a/b != 1
        print(f'done channelpruning.  {a/b}')

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


if __name__ == "__main__":
    main()
