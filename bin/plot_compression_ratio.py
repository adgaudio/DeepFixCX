"""
Show how much compression you get for varying choices of wavelet levels and
patch size.

Use the chart to choose a deepfixcx compression model that based on its spatial
resolution, scale resolution and overall compression ratio.

Usage:

    python THIS_FILE.py --patch_features l1 --input_shape 320 390
"""
import math
import torch as T
import pandas as pd
import argparse as ap
from matplotlib import pyplot as plt
import seaborn as sns
from deepfixcx.models import DeepFixCXCompression, InvalidWaveletParametersError
from deepfixcx.train import get_dset_chexpert, match, MODELS
from deepfixcx import plotting as P

p = ap.ArgumentParser()
p.add_argument('--patch_features', nargs='+', default=('l1', ))
p.add_argument('--input_shape', nargs="+", type=int, default=(320, 320))
p.add_argument('--patch_sizes', nargs='+', type=int, default=tuple())
p.add_argument('--filenameid', default='')
args = p.parse_args()
if not args.patch_sizes:
    _P_max = min(args.input_shape)//2
    _P_inc = _P_max//30 + (_P_max//30)%2
    args.patch_sizes = (
        list(range(1, 31, 2))+list(range(31, _P_max, _P_inc)) + [_P_max,2,4])
    del _P_max, _P_inc

#  dct, _ = get_dset_chexpert(train_frac=1, val_frac=0, small=True)
#  dset = dct['train_dset']

data = []
gen = ((J,P,M)
       for J in range(1, int(math.log2(min(args.input_shape)))+1)
       for P in args.patch_sizes
       for M in [1])
for J,P,M in gen:
    enc1 = DeepFixCXCompression(
        in_ch=1, in_ch_multiplier=M, levels=J,
        wavelet='db1', patch_size=P, patch_features=args.patch_features,
        how_to_error_if_input_too_small='raise')
    x = T.zeros((1,1,*args.input_shape))
    if P*P <= min(args.input_shape) / 2**J * max(args.input_shape)/2**J:
        z = enc1(x)
        cr = math.prod(z.shape) / math.prod(x.shape) * 100
        out_size = math.prod(z.shape)
    else:
        print('skip', J, P)
        cr = float('nan')
        out_size = -1
    #  plt.figure() ; plt.imshow(x.squeeze(), cmap='gray')
    #  res = enc1.wavelet_encoder(x.unsqueeze(0))
    data.append({
        'Wavelet Level, J': J, 'Patch Size, P': P, 'Channel Multiplier': M,
        'Compression Ratio (% of original size)': cr,
        'Input Shape': x.shape,
        'Output Size': out_size,
    })
df = pd.DataFrame(data)
#  print(df.to_string(float_format=lambda x: f'{x:.04f}'))

fig1, (ax1) = plt.subplots(1,1, figsize=(6,2.5), dpi=300)
#  fig1.suptitle('Compression % for varying Patch Size vs Wavelet Levels')
heatmap_data = df.pivot_table('Compression Ratio (% of original size)', 'Patch Size, P', 'Wavelet Level, J')
sns.heatmap(
    heatmap_data,
    norm=plt.cm.colors.LogNorm(), ax=ax1, annot=True, fmt='.03f', cbar=False)
#  ax1.set_title('Compression Ratio (% of original size)')

fig2, (ax2) = plt.subplots(1,1, figsize=(8,8))
sns.heatmap(
    df.pivot_table('Output Size', 'Patch Size, P', 'Wavelet Level, J'),
    norm=None, ax=ax2)
ax2.set_title('Output Size')
#  fig.tight_layout()
save_fp = f'results/plots/compression_ratio_varying_patch_and_level{args.filenameid}.png'
fig1.savefig(save_fp, bbox_inches='tight')
print('save to', save_fp)
heatmap_data.to_csv(save_fp.replace('.png', '.csv'))
fig2.savefig(f'results/plots/compression_outsize_varying_patch_and_level{args.filenameid}.png', bbox_inches='tight')
#  print(
#      df.pivot_table('Compression Ratio (% of original size)', 'Patch Size, P', 'Wavelet Level, J')
#      .to_string(float_format=lambda x: f'{x:.04f}'))

fig, ax = plt.subplots(figsize=(max(15, 1*df.shape[0]), max(15, .5*df.shape[1])))
ax.axis('off')
ax.imshow([[0]])
pd.plotting.table(
    ax,
    df.pivot_table('Compression Ratio (% of original size)', 'Patch Size, P', 'Wavelet Level, J').round(3),
    loc='center'
)
ax.set_title("Compression Ratio (% of original size)")
fig.tight_layout()
fig.savefig(f'results/plots/compression_ratio_table_as_img{args.filenameid}.png', bbox_inches='tight')

#  plt.show(block=False)
#  plt.pause(10)



#  mlp_channels, in_ch, out_ch, wavelet_levels, patch_size, mlp_depth: get_DeepFixCXEnd2End(
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
