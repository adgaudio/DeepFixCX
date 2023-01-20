# TODO: encoder throughput in 15k imgs per second.

import dataclasses as dc
import numpy as np
from simple_parsing import ArgumentParser
import time
from waveletfix.models import WaveletFixCompression
from waveletfix.train import get_dset_chexpert


def timeit(encoder, loader, device):
    secs = []
    for _ in range(5):
        s = time.time()
        N = 0
        for mb in loader:
            x = mb[0].to(device, non_blocking=False)
            N += x.shape[0]
            encoder(x)
            #  if len(secs) > 10:
                #  break
            if N > 15000:
                break
        e = time.time()
        secs.append((e-s) / N)
    return secs


@dc.dataclass
class Opts:
    """Command-line options"""
    J:int  # wavelet levels
    P:int  # patch size
    device:str = 'cpu'
    wavelet:str = 'db1'
    patch_features:str = 'l1'


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_arguments(Opts, dest='Opts')
    args = p.parse_args().Opts
    print(args)

    dct, _ = get_dset_chexpert(
        .9, .1, small=True, labels='diagnostic', epoch_size=15000)

    encoder = WaveletFixCompression(
        in_ch=1, in_ch_multiplier=1,
        levels=args.J, wavelet=args.wavelet, patch_size=args.P,
        patch_features=args.patch_features.split(',')).to(args.device)

    secs = timeit(encoder, dct['train_loader'], args.device)
    print(secs)
    print(f'J={args.J} P={args.P} device={args.device} throughput per 15k images', np.mean(secs)*15000)
