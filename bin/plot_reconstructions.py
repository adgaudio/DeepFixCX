from itertools import product
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import argparse as ap
import numpy as np
import torch as T
import torchvision.transforms as tvt

from deepfix.models import DeepFixCompression
from deepfix.models.wavelet_packet import WaveletPacket2d
from deepfix.train import get_dset_chexpert


#  rand = np.random.uniform
#  x = np.block([[rand(0,1,(50,50))+1*i*j  if i<j else np.zeros((50,50)) for i in range(4)] for j in range(3)])

p=ap.ArgumentParser()
p.add_argument('--patch_features',nargs='+',type=str,default=('l1', ))
p.add_argument('--input_shape',nargs='+',type=int,default=(320,320))
p.add_argument('--patch_sizes',nargs='+',type=int,default=tuple([1,3,5,9,19,37,79,115,160]))
p.add_argument('--wavelet',type=str,default='db1')
p.add_argument('--ssim', action='store_true')
args=p.parse_args()

# load image
dct, _ = get_dset_chexpert(train_frac=.8,val_frac=0.2,small=True)
data_loader = dct['test_loader']
img = dct['test_dset'][0][0]
C, H, W = img.shape
assert C == 1

reconstructed_imgs = [[np.ones((H, W)) for _ in range(8)]
                      for _ in args.patch_sizes]
for J,P in product(range(1,9), args.patch_sizes):
    if P > H/2**J or P > W/2**J:
        print(f"skipping  J={J} P={P}.  It doesn't do compression")
        continue
    print(f'J={J}, P={P}')
    enc = DeepFixCompression(
        in_ch=1, in_ch_multiplier=1, levels=J, wavelet=args.wavelet,
        patch_size=P, patch_features=args.patch_features,
        how_to_error_if_input_too_small='raise')
    wi = WaveletPacket2d(levels=J,wavelet=args.wavelet,inverse=True)
    # get deepfix encoding
    op = enc(img.unsqueeze(0))
    #
    # get the reconstruction
    recons = enc.reconstruct(op, (H, W), wavelet=args.wavelet, J=J, P=P)
    print(recons.max(), recons.min())
    #  recons = recons.clamp(0,1)
    #
    #
    if args.ssim:
        _, features = ssim(img.squeeze().numpy(), recons.squeeze().numpy(), full=True)
        reconstructed_imgs[args.patch_sizes.index(P)][J-1] = features
    else:
        reconstructed_imgs[args.patch_sizes.index(P)][J-1] = recons.squeeze().numpy()

# show the reconstructions with the original image overlayed at position (y,x)
fig, ax = plt.subplots()#figsize=(20,20))
y,x = 2270, 1720
imgmap = np.block(reconstructed_imgs)
#  imgmap[y-150:y+H+150, x-150:x+W+150] = 1
imgmap[y:y+H, x:x+W] = img.squeeze().numpy()
ax.imshow(imgmap, cmap='gray', interpolation='antialiased', vmin=0, vmax=1)
# styling for the overlayed original img
pady, padx = 225, 275
imgmap[y-pady:y+H+pady, x-padx:x+W+padx] = 1
imgmap[y:y+H, x:x+W] = img.squeeze().numpy()
ax.text(x+W/2, y-75, 'Original Image', horizontalalignment='center')
ax.add_patch(Rectangle(
    (x-padx,y-pady-50),W+padx*2,H+pady*2, edgecolor='gray', facecolor='none', lw=2))
# make the x and y labels and ticks show wavelet level and patch size
ax.set_yticks(
    np.linspace(H//2, H*len(args.patch_sizes)-H//2, len(args.patch_sizes)),
    labels=args.patch_sizes)
ax.set_ylabel('Patch Size, P')
ax.set_xticks(np.linspace(W//2, W*8-W//2, 8), labels=np.arange(1, 9))
ax.set_xlabel('Wavelet Level, J')
ax.set_title('DeepFix Image Reconstruction')
#  fig.suptitle('Reconstructed images')
save_fp = f'results/plots/reconstructed_imgs_{",".join(x for x in args.patch_features)}{"_ssim" if args.ssim else ""}.png'
fig.savefig(save_fp, bbox_inches='tight', dpi=300)
print('saved to:', save_fp)

plt.show()
