from itertools import product
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse as ap
import numpy as np
import pandas as pd
import seaborn as sns
import torch as T
import torchvision.transforms as tvt

from waveletfix.models import WaveletFixCompression
from waveletfix.models.wavelet_packet import WaveletPacket2d
from waveletfix.train import (
    get_dset_chexpert, get_dset_intel_mobileodt, get_dset_kimeye,
    get_dset_flowers102, get_dset_food101)


p=ap.ArgumentParser()
p.add_argument('--patch_features',nargs='+',type=str,default=('l1', ))
p.add_argument('--patch_sizes',nargs='+',type=int,default=tuple([1,3,5,9,19,37,79,115,160]))
p.add_argument('--wavelet',nargs="+",type=str,default='db1')
p.add_argument('--device', default='cuda')
p.add_argument('--overwrite', action='store_true')
p.add_argument('--dataset', default='chexpert', choices=('chexpert', 'intelmobileodt', 'kimeye', 'flowers102', 'food101'))
args=p.parse_args()

if args.overwrite:
    # load image
    if args.dataset == 'chexpert':
        dct, _ = get_dset_chexpert(train_frac=.8,val_frac=0.2,small=True)
    elif args.dataset == 'intelmobileodt':
        #  --dset intel_mobileodt:train:val:test:v1
        dct, _ = get_dset_intel_mobileodt(stage_trainval='train', use_val='val', stage_test='test', augment='v1')
    elif args.dataset == 'kimeye':
        dct, _ = get_dset_kimeye(train_frac=.7, test_frac=.15)
    elif args.dataset == 'food101':
        dct, _ = get_dset_food101()
    elif args.dataset == 'flowers102':
        dct, _ = get_dset_flowers102()
    else:
        raise NotImplemented(args.dataset)

    #  data_loader = dct['train_loader']
    img = dct['test_dset'][1][0]
    data_loader = dct['test_loader']
    ssim_store = []
    for J,P in product(range(1,9), args.patch_sizes):
        print(f'J={J}, P={P}')
        ssim_per_img = []
        enc = WaveletFixCompression(
            in_ch=1, in_ch_multiplier=1, levels=J, wavelet=args.wavelet,
            patch_size=P, patch_features=args.patch_features,
            how_to_error_if_input_too_small='raise'
        ).to(args.device, non_blocking=True)
        wi = WaveletPacket2d(levels=J,wavelet=args.wavelet,inverse=True
                             ).to(args.device, non_blocking=True)
        for n,(x,y) in enumerate(data_loader):
            B,C,H,W = x.shape
            # if P > H/2**J or P > W/2**J:
            if P*P > H / 2**J * W/2**J:
                print(f"skipping  J={J} P={P}.  It doesn't do compression")
                break
            # get waveletfix encoding
            op = enc(x.to(args.device))
            # get the reconstruction

            recons = enc.reconstruct(op, (H, W), args.wavelet, J=J, P=P)
            recons = recons.clamp(0,1)
            recons = recons.cpu()
            #
            # compute ssim for each img in the minibatch
            for im1, im2 in zip(x.squeeze(1).unbind(0), recons.squeeze(1).unbind(0)):
                _val, _map = ssim(im1.cpu().numpy(), im2.cpu().numpy(), win_size=3, full=True)
                ssim_per_img.append(_val)
                #  fig, axs = plt.subplots(1,3)
                #  axs[0].imshow(im1.numpy(), 'gray')
                #  axs[1].imshow(im2.numpy(), 'gray')
                #  axs[2].imshow(_map, 'gray')
                #  fig.suptitle(f'J={J} P={P}')
        ssim_store.append({
            'Wavelet Level, J': J, 'Patch Size, P': P,
            'Avg SSIM': np.mean(ssim_per_img)})

    df = pd.DataFrame(ssim_store)
    print(df)

    # generate plot
    pivot_table = df.pivot_table('Avg SSIM', 'Patch Size, P', 'Wavelet Level, J')
else:
    pivot_table = pd.read_csv(f'results/plots/heatmap_reconstruction_{args.dataset}_{",".join(args.patch_features)}.csv').set_index('Patch Size, P')#.drop(columns='Unnamed: 0')
fig,axs = plt.subplots(1,1, figsize=(6, 2.5), dpi=300)
sns.heatmap(
    pivot_table,
    cmap='copper',
    #  norm=plt.cm.colors.PowerNorm(2),
    ax=axs, annot=True, fmt='.03f', cbar=False)
#  axs.set_title('Privacy: Reconstruction Score')
axs.set_xlabel('Wavelet Level, J')
# save plot
save_fp = f'results/plots/heatmap_reconstruction_{args.dataset}_{",".join(args.patch_features)}.png'
fig.savefig(save_fp,bbox_inches='tight', dpi=300)
pivot_table.to_csv(save_fp.replace('.png', '.csv'))
print(f'saved to: {save_fp}')
