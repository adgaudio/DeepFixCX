import seaborn as sns
import pandas as pd
from deepfix.train import get_dset_chexpert
from deepfix import plotting as P
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from deepfix.models.wavelet_packet import WaveletPacket2d
from deepfix.models import DeepFixCompression,InvalidWaveletParametersError
import argparse as ap
import torch as T
import numpy as np
from matplotlib import pyplot as plt


p=ap.ArgumentParser()
p.add_argument('--patch_features',nargs='+',type=str,default='l1')
p.add_argument('--input_shape',nargs='+',type=int,default=(320,320))
p.add_argument('--patch_sizes',nargs='+',type=int,default=tuple())
p.add_argument('--wavelet',nargs="+",type=str,default='db1')
args=p.parse_args()

if not args.patch_sizes:
    _P_max=min(args.input_shape)//2
    _P_inc=_P_max//30 + (_P_max//30)%2
    args.patch_sizes=(list(range(1,31,2))+list(range(31,_P_max,_P_inc))+[_P_max,2,4])
    del _P_max, _P_inc

dct,_=get_dset_chexpert(train_frac=.8,val_frac=0.2,small=True)
data_loader=dct['test_loader']
args.patch_sizes=args.patch_sizes[1:]
ssim_store=[]


gen=((J,P,M) for J in range(1,9+1) for P in args.patch_sizes for M in [1])
for J,P,M in gen:
        print(J,P,M)
        si=0 
        total_batch=0
        enc=DeepFixCompression(in_ch=1, in_ch_multiplier=M, levels=J,wavelet=args.wavelet, patch_size=P, patch_features=args.patch_features,how_to_error_if_input_too_small='raise')
        wi=WaveletPacket2d(levels=J,wavelet=args.wavelet,inverse=True)
        for n,(x,y) in enumerate(data_loader):
            try:
                op=enc(x)
            except InvalidWaveletParametersError:
                si='nan'
                break
            B=x.shape[0]
            total_batch+=1
            recons=wi(op.reshape(B,M,4**J,P,P))
            mx=T.max(recons)
            mn=T.min(recons)
            recons=(recons-mn)/(mx-mn)
            x_new=np.squeeze(x.detach().numpy(),1).transpose(1,2,0)
            recons_new=np.squeeze(recons.detach().numpy(),1).transpose(1,2,0)
            si+=ssim(resize(x_new,(recons_new.shape[0],recons_new.shape[1])),recons_new,win_size=3,channel_sxis=2,multichannel=True)

        if(si!='nan'):
            si=(si/total_batch)*100
        #print(si)
        ssim_store.append({'Wavelet Levels': J, 'Patch Size': P, 'Channel Multiplier': M,'Similarity Index (%)': si})


df = pd.DataFrame(ssim_store)

fig,axs = plt.subplots(1,1, figsize=(8,4))
sns.heatmap(df.pivot_table('Similarity Index (%)', 'Patch Size', 'Wavelet Levels'),norm=plt.cm.colors.LogNorm(), ax=axs, annot=True, fmt='.03f')
axs.set_title('Structural Similarity Index (%)')
fig.savefig('results/plots/reconstruction_heatmap.png',bbox_inches='tight')
