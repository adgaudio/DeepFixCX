"""
Boilerplate to implement training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets
"""
from os.path import exists
from efficientnet_pytorch import EfficientNet
import pampy
from simple_parsing import ArgumentParser, choice
from simplepytorch import datasets as D
from simplepytorch import trainlib as TL
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Union, Callable
import dataclasses as dc
import numpy as np
import torch as T
import torchvision.models as tvm
import torchvision.transforms as tvt

from deepfix.models import effnetv2_s
from deepfix.weight_saliency import reinitialize_least_salient, costfn_multiclass
from deepfix.init_from_distribution import init_from_hist_, init_from_beta
from deepfix import deepfix_strategies as dfs


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
}

LOSS_FNS = {
    ('BCEWithLogitsLoss', ): lambda _: T.nn.BCEWithLogitsLoss(),
    ('CrossEntropyLoss', ): lambda _: T.nn.CrossEntropyLoss(),
    ('CE_intelmobileodt', ): lambda _: loss_intelmobileodt,
}

DSETS = {
    ('intel_mobileodt', str, str, str, str): (
        lambda train, val, test, aug: get_dset_intel_mobileodt(train, val, test, aug)),
}


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

def get_effnetv2(pretraining, in_channels, out_channels):
    assert pretraining == 'untrained', 'error: no pre-trained weights currently available for EfficientNetV2'
    mdl = effnetv2_s(num_classes=out_channels)
    _modify_conv2d(conv2d=mdl.features[0][0], in_channels=in_channels)
    return mdl


def get_resnet(name, pretraining, in_channels, out_channels):
    assert pretraining in {'untrained', 'imagenet'}
    mdl = getattr(tvm, name)(
        pretrained=True if pretraining == 'imagenet' else False)
    _modify_conv2d(mdl.conv1, in_channels)
    mdl.fc = T.nn.Linear(
        in_features=mdl.fc.in_features, out_features=out_channels, bias=True)
    return mdl


def get_efficientnetv1(name, pretraining, in_channels, out_channels):
    assert pretraining in {'untrained', 'imagenet', 'imagenetadv'}
    if pretraining == 'imagenetadv':
        mdl = EfficientNet.from_pretrained(
            name, advprop=True, in_channels=in_channels, num_classes=out_channels)
    elif pretraining == 'imagenet':
        mdl = EfficientNet.from_pretrained(
            name, in_channels=in_channels, num_classes=out_channels)
    else:
        mdl = EfficientNet.from_name(
            name, in_channels=in_channels, num_classes=out_channels)
    return mdl


def _modify_conv2d(conv2d:T.nn.Module, in_channels:int, ):
    """Inplace modify conv2d layer to ensure has in_channels"""
    if in_channels != conv2d.in_channels:
        conv2d.in_channels = in_channels
        if in_channels < conv2d.in_channels:
            conv2d.weight = T.nn.Parameter(conv2d.weight.data[:,[1],:,:])
        else:
            raise NotImplementedError('code for this written but not tested')
            O,_,H,W = conv2d.weight.shape
            tmp = T.empty(
                (O,in_channels,H,W),
                dtype=conv2d.weight.dtype, device=conv2d.weight.device)
            T.nn.init.kaiming_uniform_(tmp)
            conv2d.weight = T.nn.Parameter(tmp)
        assert conv2d.bias is None, 'bias not implemented yet'


def onehot(y, nclasses):
    return T.zeros((y.numel(), nclasses), dtype=y.dtype, device=y.device)\
            .scatter_(1, y.unsqueeze(1), 1)


def get_dset_intel_mobileodt(stage_trainval:str, use_val:str, stage_test:str, augment:str
                             ) -> (dict[str,Union[Dataset,DataLoader]], tuple[str]):
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


def match(spec:str, dct:dict):
    return pampy.match(spec.split(':'), *(x for y in dct.items() for x in y))


def get_model_opt(model_spec:str, opt_spec:str, device:str
                  ) -> dict[str, Union[T.nn.Module, T.optim.Optimizer]]:
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
    # optimizer
    spec = opt_spec.split(':')
    kls = getattr(T.optim, spec[0])
    params = [(x,float(y)) for x,y in [kv.split('=') for kv in spec[1:]]]
    return dict(model=mdl, optimizer=kls(mdl.parameters(), **dict(params)))


def get_lossfn(loss_spec:str) -> T.nn.Module:
    return match(loss_spec, LOSS_FNS)


def get_dset_loaders_resultfactory(dset_spec:str) -> dict:
    dct, class_names = match(dset_spec, DSETS)
    #  dct['result_factory'] = lambda: TL.MultiLabelBinaryClassification(
            #  class_names, binarize_fn=lambda yh: (T.sigmoid(yh)>.5).long())
    dct['result_factory'] = lambda: TL.MultiClassClassification(
            len(class_names), binarize_fn=lambda yh: yh.softmax(1).argmax(1))

    return dct


def get_deepfix_train_strategy(deepfix_spec:str):
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
        return dfs.DeepFix_LambdaInit(init_from_beta, args=(float(alpha), float(beta)))
    else:
        raise NotImplementedError(deepfix_spec)


def train_config(args:'TrainOptions') -> TL.TrainConfig:
    return TL.TrainConfig(
        **get_model_opt(args.model, args.opt, args.device),
        loss_fn=get_lossfn(args.lossfn),
        **get_dset_loaders_resultfactory(args.dset),
        device=args.device,
        epochs=args.epochs,
        train_one_epoch=get_deepfix_train_strategy(args.deepfix),
        experiment_id=args.experiment_id,
    )


@dc.dataclass
class TrainOptions:
    """High-level configuration for training PyTorch models
    on the IntelMobileODTCervical dataset.
    """
    epochs:int = 50
    device:str = 'cuda' if T.cuda.is_available() else 'cpu'
    dset:str = choice(
        'intel_mobileodt:train:val:test:v1',
        'intel_mobileodt:train+additional:val:test:v1',
        'intel_mobileodt:train+additional:noval:test:v1',
        default='intel_mobileodt:train:val:test:v1')
    opt:str = 'SGD:lr=.001:momentum=.9:nesterov=1'
    lossfn:str = choice(*(x[0] for x in LOSS_FNS.keys()), default='CrossEntropyLoss')
    model:str = 'resnet18:imagenet:3:3'  # Model specification adheres to the template "model_name:pretraining:in_ch:out_ch"
    deepfix:str = 'off'  # DeepFix Re-initialization Method.
                         #  "off" or "reinit:N:P:R" or "d[f]hist:path_to_histogram.pth"
                         #  or "beta:A:B" for A,B as (float) parameters of the beta distribution
    experiment_id:str = 'debugging'
    init:str = 'unchanged'  # 'unchanged' or 'dhist:path_to_histogram.pth'

    def execute(self):
        cfg = train_config(self)
        cfg.train(cfg)


def main():
    p = ArgumentParser()
    p.add_arguments(TrainOptions, dest='TrainOptions')
    args = p.parse_args().TrainOptions
    print(args)
    cfg = train_config(args)
    cfg.train(cfg)


if __name__ == "__main__":
    main()
