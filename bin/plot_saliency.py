from collections import defaultdict
from itertools import product
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple
import captum.attr
import dataclasses as dc
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import scipy.cluster.hierarchy as sph
import torch as T
from waveletfix.train import get_dset_chexpert
from waveletfix.models.wavelet_packet import WaveletPacket2d
from waveletfix import plotting as dplt


if __name__ == "__main__":
    @dc.dataclass
    class CmdlineOpts:
        J:int = 2
        P:int = 19
        device = 'cuda'
        def __post_init__(self):
            self.save_dir = f'results/plots/saliency/J={self.J}.P={self.P}'
    from simple_parsing import ArgumentParser
    args = ArgumentParser()
    args.add_arguments(CmdlineOpts, dest='args')
    args = args.parse_args().args

    # create a directory to save images
    os.makedirs(args.save_dir, exist_ok=True)

    # "static" config
    J, P = args.J, args.P
    wavelet='db1'
    fp = f'results/2.C21.J={J}.P={P}/checkpoints/epoch_80.pth'
    assert wavelet == 'db1' and '2.C21' in fp, 'setup error'
    orig_img_shape = (320, 320)  # original image size (320x320 imgs)
    B = int(os.environ.get('batch_size', 15))
    I = 1  # num input image channels (1 for x-ray)

    # get model
    print("get model and dataloader")
    mdl = T.load(fp, map_location=args.device)['model'].eval()
    # backwards compatibility with earlier version of waveletfix
    mdl.compression_mdl.wavelet_encoder.adaptive = 0

    # get dataloader
    dset_dct, class_names = get_dset_chexpert(.9, .1, small=True)
    d = dset_dct['test_dset']

    # get the optimal thresholds from ROC AUC computed on validation set
    class_thresholds = pd.read_csv(
        f'{os.path.dirname(os.path.dirname(fp))}/class_thresholds.csv',
        index_col=0)['0'].reindex(class_names)
    assert class_thresholds.isnull().sum() == 0


    # generate saliency plots for all images
    num_imgs_so_far = 0
    print("Compute saliency")
    for mb in dset_dct['test_loader']:
        x = mb[0].to(args.device, non_blocking=True)
        _B, _C = x.shape[:2]
        num_imgs_so_far += _B
        with T.no_grad():
            enc = mdl.compression_mdl(x)
            yhat = mdl.mlp(enc).cpu()
            yhat2 = (yhat.sigmoid() > T.tensor(class_thresholds.values.reshape(1,-1))).to(T.int8)

        recon_img = mdl.compression_mdl.reconstruct(
            enc, orig_img_shape, wavelet=wavelet, J=J, P=P).cpu()

        # TODO: continue from here to generate saliency plots.
        # each title should say TP or FP on it.
        # one image for each class.
        #  fig.savefig("{args.save_dir}/{idx}_{class_idx}.png", bbox_inches='tight', dpi=300)

        #  explainer = captum.attr.IntegratedGradients(mdl.mlp)
        #  explainer = captum.attr.NoiseTunnel(captum.attr.Saliency( T.nn.Sequential(mdl.mlp, )))
        #  explainer = captum.attr.NoiseTunnel(captum.attr.DeepLift(mdl.mlp, True))
        explainer = captum.attr.DeepLift(mdl.mlp, True)
        #
        baseline = enc.reshape(_B, 4**J,P,P).clone()
        baseline[:,1:] = 0
        baseline = baseline.reshape(enc.shape)
        #
        # generate attribution plot for each image, for each class
        #  for class_name in class_names:
        for class_name in ['Pleural Effusion']:
            # get attributions for each class
            #
            class_idx = class_names.index(class_name)
            posthoc_explanation_enc = explainer.attribute(
                    enc.requires_grad_(True), target=class_names.index('Pleural Effusion'),
                    baselines=baseline,
                    #  nt_samples=50, nt_samples_batch_size=B, nt_type='smoothgrad',
                ).detach().reshape((_B, I, 4**J, P, P)).float()
            posthoc_explanation_img = mdl.compression_mdl.reconstruct(
                posthoc_explanation_enc, orig_img_shape, wavelet=wavelet, J=J, P=P,
                restore_orig_size=True).cpu()
            for idx in range(_B):
                arbitrary_img_id = num_imgs_so_far - _B + idx
                #
                # ... figure out if prediction for this class and image was TP, FP, etc
                cmcell = {  # (y, yhat): 'square of confusion matrix'
                    (1,1): 'TP', (0,0): 'TN', (1,0): 'FN', (0,1): 'FP'
                }[(mb[1][idx, class_idx].item(), yhat2[idx, class_idx].item())]
                #
                # ... prep the visual
                a = posthoc_explanation_img.abs().permute(0,2,3,1).cpu().numpy()
                b = T.nn.functional.interpolate(
                    x, posthoc_explanation_img.shape[-2:]).permute(0,2,3,1).cpu().numpy()
                #
                # ... plot it
                fig, (ax1, ax2) = plt.subplots(1,2, num=1, clear=True)
                ax1.imshow(mb[0][idx].squeeze().numpy(), 'Greys_r')
                captum.attr.visualization.visualize_image_attr(
                    #  a[idx], b[idx], 'blended_heat_map', alpha_overlay=.6,
                    a[idx], recon_img[idx].permute(1,2,0).numpy(), 'blended_heat_map', alpha_overlay=.6,
                    plt_fig_axis=(None, ax2),
                )
                # ... titles and stuff
                ax1.set_title(f'Original image')
                ax2.set_title('Attribution, ' + cmcell)
                ax1.axis('off')
                ax2.axis('off')
                # ... save figure
                os.makedirs(f'{args.save_dir}/{class_name}/', exist_ok=True)
                save_fp = f'{args.save_dir}/{class_name}/{cmcell}_{arbitrary_img_id}.png'
                fig.tight_layout()
                fig.subplots_adjust(hspace=0, wspace=0)
                fig.savefig(save_fp, bbox_inches='tight')
                print(save_fp)
                #  break
            #  break
        #  break
