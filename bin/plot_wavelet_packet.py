from deepfix import plotting as P
from deepfix.models.wavelet_packet import WaveletPacket2d
from matplotlib import pyplot as plt
import numpy as np
import pywt
import seaborn as sns
import skimage.data
import torch as T


plt.ion()


def plot_wavelet_packet_transform():
    # show different wavelets
    im = T.tensor(plt.imread('data/x-ray.jpg')/255)
    #  im -= im.min()
    J = 2
    plt.figure();plt.imshow(im, cmap='gray')
    for wavelet in ['bior3.1', 'db1', 'coif2']:
        wp = WaveletPacket2d(wavelet, J)
        res = wp(im.unsqueeze(0).unsqueeze(0).float())
        z = [x.clamp(x.quantile(.05), x.quantile(.95)) for x in res.squeeze().unbind()]
        fig = P.plot_img_grid(
            z,
            cmap='PRGn',
            norm=[ plt.cm.colors.CenteredNorm(0) ] + [plt.cm.colors.TwoSlopeNorm(
                0, min(x.min() for x in z[1:]).item(), max(x.max() for x in z[1:]).item())
            ]*(len(z)-1),
            ax_titles=[a+b for a in 'AVHD' for b in 'AVHD']
        )#, norm=plt.cm.colors.SymLogNorm(.1))
        save_fp = f'./results/plots/wp_{wavelet}_J{J}.png'
        print('save to: ', save_fp)
        fig.savefig(save_fp, bbox_inches='tight')


def plot_haar_wavelet_filters():
    lo,hi = pywt.Wavelet('db1').filter_bank[:2]  # same as haar and bior1.1 etc
    fig, axs = plt.subplots(2,2, figsize=(4,4))
    kws = dict(
        cmap='PRGn', norm=plt.cm.colors.CenteredNorm(0), cbar=False,
        annot=True, annot_kws=dict(fontsize='xx-large'))
    sns.heatmap(np.outer(lo, lo), ax=axs[0,0], **kws)
    sns.heatmap(np.outer(lo, hi), ax=axs[0,1], **kws)
    sns.heatmap(np.outer(hi, lo), ax=axs[1,0], **kws)
    sns.heatmap(np.outer(hi, hi), ax=axs[1,1], **kws)
    axs[0,0].set_ylabel(
        r'$\left[ \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right]$', fontsize=23, rotation=-90, va='top')
    axs[1,0].set_ylabel(
        r'$\left[ \frac{-1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right]$', fontsize=23, rotation=-90, va='top')
    axs[0,0].set_title(
        r'$\left[ \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right]$', fontsize=23)
    axs[0,1].set_title(
        r'$\left[ \frac{-1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right]$', fontsize=23)
    [ax.set_yticks([]) for ax in axs.ravel()]
    [ax.set_xticks([]) for ax in axs.ravel()]
    fig.tight_layout()
    save_fp = f'./results/plots/wp_haar_filters_2d.png'
    print('save to: ', save_fp)
    fig.savefig(save_fp, bbox_inches='tight', dpi=300)



if __name__ == "__main__":
    plot_wavelet_packet_transform()
    plot_haar_wavelet_filters()
    plt.show()
    plt.pause(5)
