import torch as T
import dataclasses as dc
from simple_parsing import ArgumentParser
from typing import List
from deepfix.plotting import plot_img_grid
from deepfix.models import DeepFixCompression
from simplepytorch.datasets import CheXpert_Small
import torchvision.transforms as tvt
from matplotlib import pyplot as plt
import numpy as np
from welford import Welford
from os.path import dirname
import os


@dc.dataclass
class Options:
    level:int = 1
    patchsize:int = 80
    wavelet:str = 'coif1'
    patch_features:List[str] = ('l1', )
    device:str = 'cuda'
    overwrite_existing:bool = False
    zero_mean:bool = False
    dset:str = 'chexpert_small'
    def __post_init__(self):
        self.savefp = f'norms/{self.dset}:{self.wavelet}:{self.level}:{self.patchsize}:{",".join(self.patch_features)}:{int(self.zero_mean)}.pth'


def parse_args(argv=None) -> Options:
    par = ArgumentParser()
    par.add_arguments(Options, 'args')
    return par.parse_args(argv).args


def get_model_and_dset(args):
    # model
    deepfix_mdl = DeepFixCompression(
        in_ch=1, in_ch_multiplier=1,
        levels=args.level, wavelet=args.wavelet,
        patch_size=args.patchsize, patch_features=args.patch_features,
        how_to_error_if_input_too_small='warn')
    deepfix_mdl.to(args.device)
    # dataset:  chexpert dataset
    assert args.dset == 'chexpert_small', 'not implemented'
    dset = CheXpert_Small(
        use_train_set=True,
        img_transform=tvt.Compose([
            tvt.RandomCrop((320, 320)),
            tvt.ToTensor(),  # full res 1024x1024 imgs
        ]),
        getitem_transform=lambda dct: dct)
    return deepfix_mdl, dset


def get_deepfixed_img_and_labels(deepfix_model, dset, idx, device):
    dct = dset[idx]
    x = dct['image'].to(device, non_blocking=True)
    patient_id = dct['labels'].loc['Patient']
    x_deepfix = deepfix_model(x.unsqueeze(0))
    metadata = {'labels': dct['labels'], 'fp': dct['fp'],
                'filesize': x.shape, 'compressed_size': x_deepfix.shape}
    return x_deepfix, patient_id, metadata


if __name__ == "__main__":
    import sys
    args = parse_args()
    print(args)
    deepfix_mdl, dset = get_model_and_dset(args)

    if os.path.exists(args.savefp) and args.overwrite_existing is False:
        print("Data already exists.  Not overwriting it.  Bye!")
        sys.exit()

    # get mean and var of whole training set.
    streaming_stats = Welford()
    #  tst = []
    #  for i in [0,1,2,3,4]:
    #  for i in range(1000):
    for i in range(len(dset)):
        img = dset[i]['image'].unsqueeze(0).to(args.device)
        enc = deepfix_mdl(img)
        streaming_stats.add(enc.cpu().double().numpy())
        #  tst.append(enc)

        #  enc = enc.reshape(
            #  4**args.level*len(args.patch_features),
            #  args.patchsize, args.patchsize)
    dct = {
        'means': T.tensor(streaming_stats.mean, dtype=T.float),
        'vars': T.tensor(streaming_stats.var_p, dtype=T.float)}
    os.makedirs(dirname(args.savefp), exist_ok=True)
    T.save(dct, args.savefp)
    print(f'Wrote fp:  {args.savefp}')

    # passes test?
    #  mu_r = T.stack(tst).mean(0)
    #  T.allclose(mu_r.cpu(), dct['means'], 1e-7)
    #  var_r = T.stack(tst).var(0, unbiased=False)
    #  T.allclose(var_p.cpu(), dct['vars'], 1e-6)



    # analyze a batch of images
    #  batch = [dset[i]['image'].to(args.device) for i in range(20)]
    #  batch = [x - x.mean() for x in batch]
    #  batch = T.stack(batch)
    #  enc = deepfix_mdl(batch)
    #  print(enc.shape)
    #  import sys ; sys.exit()
    #  plt.plot(min_, label='min')
    #  plt.plot(l1, label='l1')


        #  enc_log4 = enc.log2()/enc.new_tensor(4).log2()

        #  plt.subplots() ; plt.scatter(enc.var((-1,-2)).cpu(), enc.mean((-1,-2)).cpu())
        #  plt.subplots() ; plt.plot(enc.view(-1).cpu())
        #  fig, axs = plt.subplots(2,1) ; axs[0].plot(enc.log2().cpu().view(-1)) ; axs[1].plot((enc.log2()/(enc.new_tensor(4).log2())).cpu().view(-1))
        # idea: sum l1(enc) + l1(log(enc)) gives stuff in a nice range
        #  fig.suptitle(i)

        #  plot_img_grid(enc[:100].log2()/enc.new_tensor(4).log2(), 'log4')
        #  plot_img_grid(enc[:100].log2(), 'log2')
        #  plot_img_grid(enc[:100], 'no log')
        #  plot_img_grid(enc[:100]+enc[:100].log2()/enc.new_tensor(4).log2(), 'no log + log')
