from itertools import product
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse as ap
import numpy as np
import pandas as pd
import seaborn as sns
import torch as T
import torchvision.transforms as tvt

from deepfix.models import DeepFixCompression
from deepfix.models.wavelet_packet import WaveletPacket2d
from deepfix.train import get_dset_chexpert


p=ap.ArgumentParser()
p.add_argument('--patch_features',nargs='+',type=str,default=('l1', ))
p.add_argument('--patch_sizes',nargs='+',type=int,default=tuple([1,3,5,9,19,37,79,115,160]))
p.add_argument('--wavelet',nargs="+",type=str,default='db1')
args=p.parse_args()

dct, _ = get_dset_chexpert(train_frac=.8,val_frac=0.2,small=True)
data_loader = dct['test_loader']
ssim_store = []
for J,P in product(range(1,9), args.patch_sizes):
    print(f'J={J}, P={P}')
    ssim_per_img = []
    enc = DeepFixCompression(
        in_ch=1, in_ch_multiplier=1, levels=J, wavelet=args.wavelet,
        patch_size=P, patch_features=args.patch_features,
        how_to_error_if_input_too_small='raise')
    wi = WaveletPacket2d(levels=J,wavelet=args.wavelet,inverse=True)
    for n,(x,y) in enumerate(data_loader):
        B,C,H,W = x.shape
        if P > H/2**J or P > W/2**J:
            print(f"skipping  J={J} P={P}.  It doesn't do compression")
            break
        # get deepfix encoding
        op = enc(x)
        # get the reconstruction
        repY, repX = int(np.ceil(H/2**J/P)), int(np.ceil(W/2**J/P))
        recons = wi(
            op.reshape(B,1,4**J,P,P)
            .repeat_interleave(repX, dim=-1).repeat_interleave(repY, dim=-2)
        )
        # ... restore original size by removing any padding created by deepfix
        recons = tvt.CenterCrop((H,W))(recons)
        #
        # .. normalize the image values into [0,1] (based on batch min and max)
        # because the l1 pooling makes reconstructed values outsize [0,1]
        mx = T.max(recons)
        mn = T.min(recons)
        #  print(mx, mn)
        recons = (recons-mn)/(mx-mn)
        #
        # compute ssim for each img in the minibatch
        for im1, im2 in zip(x.squeeze(1).unbind(0), recons.squeeze(1).unbind(0)):
            _val, _map = ssim(im1.numpy(), im2.numpy(), win_size=3, full=True)
            ssim_per_img.append(_val)
    ssim_store.append({
        'Wavelet Level, J': J, 'Patch Size, P': P,
        'Avg SSIM': np.mean(ssim_per_img)})

df = pd.DataFrame(ssim_store)
print(df)

# generate plot
pivot_table = df.pivot_table('Avg SSIM', 'Patch Size, P', 'Wavelet Level, J')

#  pivot_table = pd.read_csv('results/plots/heatmap_reconstruction.csv').set_index('Patch Size, P')#.drop(columns='Unnamed: 0')
fig,axs = plt.subplots(1,1)#, figsize=(8,4))
sns.heatmap(
    pivot_table,
    cmap='Blues_r',
    norm=plt.cm.colors.PowerNorm(1.5),
    ax=axs, annot=True, fmt='.03f', cbar=False)
axs.set_title('Privacy: Reconstruction Score')
# save plot
save_fp = f'results/plots/heatmap_reconstruction_{",".join(args.patch_features)}.png'
fig.savefig(save_fp,bbox_inches='tight', dpi=300)
pivot_table.to_csv(save_fp.replace('.png', '.csv'))
print(f'saved to: {save_fp}')
