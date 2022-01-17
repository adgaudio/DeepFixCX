import math
from deepfix.models import DeepFixCompression
from deepfix.train import get_dset_chexpert



dct, _ = get_dset_chexpert(train_frac=1, val_frac=0, small=True)
dset = dct['train_dset']

enc1 = DeepFixCompression(
    in_ch=1, in_ch_multiplier=1, levels=4,
    wavelet='db1', patch_size=12)
enc3 = DeepFixCompression(
    in_ch=1, in_ch_multiplier=3, levels=3,
    wavelet='db1', patch_size=3)
enc32 = DeepFixCompression(
    in_ch=1, in_ch_multiplier=3, levels=9,
    wavelet='db1', patch_size=32)
enc64 = DeepFixCompression(
    in_ch=1, in_ch_multiplier=3, levels=1,
    wavelet='coif1', patch_size=64)

from deepfix import plotting as P
from matplotlib import pyplot as plt
for J, P, M in [(7,1,1), (7,1,3), (6,3,1), (2,30,1)]:
    enc1 = DeepFixCompression(
        in_ch=1, in_ch_multiplier=M, levels=J,
        wavelet='db1', patch_size=P)
    for x,y in dset:
        print(x.numel())
        z = enc1(x.unsqueeze(0))
        #  plt.figure() ; plt.imshow(x.squeeze(), cmap='gray')
        #  res = enc1.wavelet_encoder(x.unsqueeze(0))
        print(f'J={J} P={P} M={M}')
        print(f'compression ratio: {math.prod(z.shape) / math.prod(x.shape) * 100:0.2f}%')
        print(f'array size', math.prod(z.shape), math.prod(x.shape))
        #  P.plot_img_grid(res[0].reshape(-1, *res.shape[-2:])[:200])#, norm=plt.cm.colors.SymLogNorm(.001))
        break



#  mlp_channels, in_ch, out_ch, wavelet_levels, patch_size, mlp_depth: get_DeepFixEnd2End(
# 78.7  waveletmlp:700:1:14:6:1:3
# 19.69 waveletmlp:700:1:14:7:1:3
# 78.7  waveletmlp:700:1:14:6:2:3
# 44.31 waveletmlp:700:1:14:5:3:3
# 78.9  waveletmlp:700:1:14:5:4:3
# 30.77 waveletmlp:700:1:14:4:5:3
# 44.31 waveletmlp:700:1:14:4:6:3
# 60.31 waveletmlp:700:1:14:4:7:3
# 78.77 waveletmlp:700:1:14:4:12:3

#  waveletmlp:700:1:14:4:5:3
