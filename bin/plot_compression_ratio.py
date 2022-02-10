"""
Show how much compression you get for varying choices of wavelet levels and
patch size.

Use the chart to choose a deepfix compression model that based on its spatial
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
from deepfix.models import DeepFixCompression, InvalidWaveletParametersError
from deepfix.train import get_dset_chexpert, match, MODELS
from deepfix import plotting as P

p = ap.ArgumentParser()
p.add_argument('--patch_features', nargs='+', default='l1')
p.add_argument('--input_shape', nargs="+", type=int, default=(390, 390))
args = p.parse_args()

#  dct, _ = get_dset_chexpert(train_frac=1, val_frac=0, small=True)
#  dset = dct['train_dset']

data = []
gen = ((J,P,M)
       for J in range(1, 9+1)
       for P in list(range(1,32)) + [64, 128, 256] #[1, 2, 3, 8, 16, 32, 128]
       for M in [1])
for J,P,M in gen:
    enc1 = DeepFixCompression(
        in_ch=1, in_ch_multiplier=M, levels=J,
        wavelet='db1', patch_size=P, patch_features=args.patch_features,
        how_to_error_if_input_too_small='raise')
    x = T.zeros((1,1,*args.input_shape))
    try:
        z = enc1(x)
        cr = math.prod(z.shape) / math.prod(x.shape) * 100
        out_size = math.prod(z.shape)
    except InvalidWaveletParametersError:
        cr = float('nan')
        out_size = -1
    #  plt.figure() ; plt.imshow(x.squeeze(), cmap='gray')
    #  res = enc1.wavelet_encoder(x.unsqueeze(0))
    data.append({
        'Wavelet Levels': J, 'Patch Size': P, 'Channel Multiplier': M,
        'Compression Ratio (%)': cr,
        'Input Shape': x.shape,
        'Output Size': out_size,
    })
df = pd.DataFrame(data)
print(df.to_string(float_format=lambda x: f'{x:.04f}'))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
fig.suptitle('Compression % for varying Patch Size vs Wavelet Levels')
sns.heatmap(
    df.pivot_table('Compression Ratio (%)', 'Patch Size', 'Wavelet Levels'),
    norm=plt.cm.colors.LogNorm(), ax=ax1, annot=True, fmt='.03f')
ax1.set_title('Compression Ratio (%)')
sns.heatmap(
    df.pivot_table('Output Size', 'Patch Size', 'Wavelet Levels'),
    norm=None, ax=ax2)
ax2.set_title('Output Size')
fig.tight_layout()
fig.savefig('results/plots/compression_ratio_varying_patch_and_level.png', bbox_inches='tight')
print(
    df.pivot_table('Compression Ratio (%)', 'Patch Size', 'Wavelet Levels')
    .to_string(float_format=lambda x: f'{x:.04f}'))

fig, ax = plt.subplots(figsize=(max(15, 1*df.shape[0]), max(15, .5*df.shape[1])))
ax.axis('off')
ax.imshow([[0]])
pd.plotting.table(
    ax,
    df.pivot_table('Compression Ratio (%)', 'Patch Size', 'Wavelet Levels').round(3),
    loc='center'
)
ax.set_title("Compression Ratio (%) for varying Patch Size (rows) and Wavelet Level (columns)")
fig.tight_layout()
fig.savefig('results/plots/compression_ratio_table_as_img.png', bbox_inches='tight')

plt.show(block=False)
plt.pause(10)



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
