"""
Boilerplate to implement training different networks on different datasets
with varying config.

I wish a machine could automate setting up decent baseline models and datasets
"""
from simplepytorch import trainlib as TL
from simplepytorch import datasets as D
import numpy as np
import dataclasses as dc
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from argparse_dataclass import ArgumentParser
from pampy import match, _ as ANY


MODELS = {
    ('effnetv2', str, int, int), (
        lambda _, pretrain, in_ch, out_ch: get_effnetv2(pretrain, in_ch, out_ch)),
}

LOSS_FNS = {
    ('BCELoss', ): lambda _: T.nn.BCELoss()
}

DSETS = {
    ('intel_mobileodt', str, str, str): (
        lambda _, train, val_or_test, aug: get_dset_intel_mobileodt(train, val_or_test, aug)),
}


def get_effnetv2(pretraining, in_channels, out_channels):
    #  ... TODO
    ...


def get_dset_intel_mobileodt(train:str, val_or_test:str, augment:str) -> (Dataset, Dataset, tuple[str]):
    """Obtain train/test split or random stratified train/val split for the
    IntelMobileODT Cervical Cancer Colposcopy dataset.

    Args:
        train: the `stage` for IntelMobileODTCervical(...).
            i.e. Possible choices:  {'train', 'train+additional'}
        val_or_test: one of {'val', 'test'}.
            If 'val', split out 30\% of train set.
            If 'test', use the IntelMobileODTCervical('test') test set.
        augment: Type of augmentations to apply.  One of {'v1', }
    Returns:
        (train_dset, val_dset, class_names)
            where `class_names` == ('Type 1', 'Type 2', 'Type 3')
    """
    dset_train = D.IntelMobileODTCervical(train)
    label_names = [x.replace('_', ' ') for x in dset_train.LABEL_NAMES]
    if val_or_test == 'val':   # split dset using a simple stratified split.
        y = [dset_train.getitem(i, load_img=False)
             for i in range(len(dset_train))]
        idxs_train, idxs_val = list(
            StratifiedShuffleSplit(1, test_size=.3).split(
                np.arange(len(dset_train)), y))[0]
        dset_val = T.utils.data.Subset(dset_train, idxs_val)
        dset_train = T.utils.data.Subset(dset_train, idxs_train)
    elif val_or_test == 'test':
        dset_val = D.IntelMobileODTCervical(test))
    else:
        raise NotImplementedError('Unrecognized dataset spec: {val_or_test}')
    # TODO: augmentation
    # TODO: decide whether to handle the weird data quality problems in the dataset
    return dset_train, dset_val, labels


def get_model(model_spec:str) -> T.nn.Module:
    """
    Args:
    model_spec: a string of form,
        "model_name:pretraining:in_channels:out_classes".  For example:
        "effnetv2:imagenet_pretrained:1:5"
    Returns:
        a pytorch model
    """
    return match(model_spec.split(':'), **MODELS)


def get_lossfn(loss_spec:str) -> T.nn.Module:
    return match(loss_spec.split(':'), **LOSS_FNS)


def get_dset_loaders_resultfactory(dset_spec:str) -> (Dataset, Dataset):
    spec = dset_spec.split(':')
    train_dset, val_dset, class_names = match(spec, **DSETS)
    return dict(
        train_dset=train_dset,
        val_dset=val_dset,
        train_loader=DataLoader(train_dset, batch_size=8, shuffle=True),
        val_loader=DataLoader(val_dset, batch_size=8),
        result_factory=TL.MultiLabelBinaryClassification(
            class_names, binarize_fn=lambda yh: (T.sigmoid(yh)>.5).long())
    )


def train_config(args):
    return TL.TrainConfig(
        model=get_model(args.model),
        opt=T.nn.SGD(momentum=.9),
        loss_fn=get_lossfn(args.lossfn),
        **get_dset_loaders_resultfactory(args.dset),
        device=args.device,
        epochs=args.epochs,
        checkpoint_if: Callable[['TrainConfig'], Optional[str]] = (
            lambda cfg, log_data:
            f'{cfg.base_dir}/checkpoints/epoch_{cfg.epochs}.pth'
            if log_data['epoch'] in {cfg.epochs, cfg.start_epoch} else None)
    )


@dc.dataclass
class CmdLineOptions:
    epochs:int = 20
    device:str = 'cuda' if T.cuda.is_available() else 'cpu'
    dset:str = 'intel_mobileodt:train:val:v1'
    #  dset:str = 'intel_mobileodt:train+additional:test'
    model:str = 'effnetv2:imagenet:3:1'  # todo: verify num output classes.


def main():
    args = ArgumentParser(Options).parse_args()
    cfg = train_config(args)
    cfg.train()
    # TODO: modify the train method to incorporate re-initialization
    # TODO: dataset augmentation and pre-processing
    # TODO: effnetv2 mdl.


if __name__ == "__main__":
    main()
