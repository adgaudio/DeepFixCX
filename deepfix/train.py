"""
Boilerplate to implement training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets
"""
from argparse_dataclass import ArgumentParser
from pampy import match
from simplepytorch import datasets as D
from simplepytorch import trainlib as TL
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Union
import dataclasses as dc
import numpy as np
import torch as T
import torchvision.transforms as tvt

from deepfix.models import effnetv2_s


MODELS = {
    ('effnetv2', str, str, str): (
        lambda pretrain, in_ch, out_ch: get_effnetv2(pretrain, int(in_ch), int(out_ch))),
}

LOSS_FNS = {
    ('BCEWithLogitsLoss', ): lambda _: T.nn.BCEWithLogitsLoss()
}

DSETS = {
    ('intel_mobileodt', str, str, str): (
        lambda train, test, aug: get_dset_intel_mobileodt(train, test, aug)),
}


def get_effnetv2(pretraining, in_channels, out_channels):
    assert pretraining == 'untrained', 'error: no pre-trained weights currently available for EfficientNetV2'
    mdl = effnetv2_s(num_classes=3)
    if in_channels != 3:
        conv2d = mdl.features[0][0]
        conv2d.in_channels = in_channels
        if in_channels < 3:
            conv2d.weight = T.nn.Parameter(conv2d.weight.data[:,[1],:,:])
        else:
            raise NotImplementedError('code for this written but not tested')
            O,_,H,W = conv2d.weight.shape
            tmp = T.empty(
                (O,in_channels,H,W),
                dtype=conv2d.weight.dtype, device=conv2d.weight.device)
            T.nn.init.kaiming_uniform_(tmp)
            conv2d.weight = T.nn.Parameter(tmp)
        assert conv2d.bias is None
    return mdl


def onehot(y, nclasses):
    return T.zeros((y.numel(), nclasses), dtype=y.dtype, device=y.device)\
            .scatter_(1, y.unsqueeze(1), 1)


def get_dset_intel_mobileodt(stage_trainval:str, stage_test:str, augment:str
                             ) -> (dict[str,Union[Dataset,DataLoader]], tuple[str]):
    """Obtain train/val/test splits for the IntelMobileODT Cervical Cancer
    Colposcopy dataset, and the data loaders.

    Args:
        stage_trainval: the `stage` for training and validation.
            i.e. Possible choices:  {'train', 'train+additional'}
            Train / val split is 70/30 random stratified split.
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
    idxs_train, idxs_val = list(
        StratifiedShuffleSplit(1, test_size=.3).split(
            np.arange(len(dset_trainval)), _y))[0]
    dct = dict(
        train_dset=T.utils.data.Subset(dset_trainval, idxs_train),
        val_dset=T.utils.data.Subset(dset_trainval, idxs_val),
        test_dset=D.IntelMobileODTCervical(stage_test, base_dir))

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
        onehot(xy[1].unsqueeze(0).long()-1, 3).squeeze_().float()))
        for k,v in dct.items()}
    dct.update(dict(
        train_loader=DataLoader(dct['train_dset'], batch_size=20, shuffle=True),
        val_loader=DataLoader(dct['val_dset'], batch_size=20),
        test_loader=DataLoader(dct['test_dset'], batch_size=20),))
    class_names = [x.replace('_', ' ') for x in D.IntelMobileODTCervical.LABEL_NAMES]
    return dct, class_names


def get_model_opt(model_spec:str, opt_spec:str, device:str
                  ) -> dict[str, Union[T.nn.Module, T.optim.Optimizer]]:
    """
    Args:
        model_spec: a string of form,
            "model_name:pretraining:in_channels:out_classes".  For example:
            "effnetv2:imagenet_pretrained:1:5"
        opt_spec: Specifies how to create optimizer.
            First value is a pytorch Optimizer in T.optim.*.
            Other values are numerical parameters.
            Example: "SGD:lr=.001:momentum=.9"
        device: e.g. 'cpu' or 'gpu'
    Returns:
        a pytorch model and optimizer
    """
    mdl = match(model_spec.split(':'), *(x for y in MODELS.items() for x in y))
    mdl = mdl.to(device, non_blocking=True)
    # optimizer
    spec = opt_spec.split(':')
    kls = getattr(T.optim, spec[0])
    params = [(x,float(y)) for x,y in [kv.split('=') for kv in spec[1:]]]
    return dict(model=mdl, optimizer=kls(mdl.parameters(), **dict(params)))


def get_lossfn(loss_spec:str) -> T.nn.Module:
    return match(loss_spec.split(':'), *(x for y in LOSS_FNS.items() for x in y))


def get_dset_loaders_resultfactory(dset_spec:str) -> dict:
    dct, class_names = match(
        dset_spec.split(':'), *(x for y in DSETS.items() for x in y))
    dct['result_factory'] = lambda: TL.MultiLabelBinaryClassification(
            class_names, binarize_fn=lambda yh: (T.sigmoid(yh)>.5).long())
    return dct


def train_config(args):
    return TL.TrainConfig(
        **get_model_opt(args['model'], args['opt'], args['device']),
        loss_fn=get_lossfn(args['lossfn']),
        **get_dset_loaders_resultfactory(args['dset']),
        device=args['device'],
        epochs=args['epochs'],
        checkpoint_if = (
            lambda cfg, log_data:
            f'{cfg.base_dir}/checkpoints/epoch_{cfg.epochs}.pth'
            if log_data['epoch'] in {cfg.epochs, cfg.start_epoch} else None)
    )


@dc.dataclass
class CmdLineOptions:
    epochs:int = 20
    device:str = 'cuda' if T.cuda.is_available() else 'cpu'
    dset:str = 'intel_mobileodt:train:test:v1'
    opt:str = 'SGD:lr=.001:momentum=.9'
    lossfn:str = 'BCEWithLogitsLoss'
    #  dset:str = 'intel_mobileodt:train+additional:test'
    model:str = 'effnetv2:untrained:3:1'  # todo: verify num output classes.


def main():
    args = ArgumentParser(CmdLineOptions).parse_args().__dict__
    cfg = train_config(args)
    cfg.train(cfg)
    # TODO: modify the train_one_epoch method to incorporate re-initialization
    # TODO: dataset augmentation: fix cross dataset contamination by hashing imgs


if __name__ == "__main__":
    main()
