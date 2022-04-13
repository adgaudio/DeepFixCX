#!/usr/bin/env python
"""
Get the anonymity score for a deepfix encoder model.

python bin/anonymity_score.py --dset chexpert_small:.1:.001 --model waveletmlp:700:1:14:7:32:3:3 --lossfn chexpert_uignore
"""
import shutil
from simple_parsing import ArgumentParser
import dataclasses as dc
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import pandas as pd
import seaborn as sns
from os.path import dirname
from os import makedirs
import torch as T
import torchvision.transforms as tvt
import scipy.stats
import scipy.spatial.distance

from simplepytorch.datasets import CheXpert_Small
from deepfix.models import DeepFixCompression
from deepfix.models.waveletmlp import Normalization


def euclidean_dist(vec1, vec2):
    """Compute B Distances between two tensors of shape (B, ...), returning a pairwise distance matrix"""
    B = vec1.shape[0]
    assert vec1.shape[0] == vec2.shape[0]
    return ((vec1.reshape(B, 1, -1) - vec2.reshape(1, B, -1))**2).sum((2)).sqrt()


@dc.dataclass
class Options:
    n_patients: int = 5  # choose as high as fits in ram.
    level: int = 2
    patchsize: int = 64
    wavelet: str = 'db1'
    patch_features: List[str] = ('l1', )
    device: str = 'cuda'
    save_fp: str = './results/anonymity_scores/{experiment_id}.pth'
    save_img_fp: str = './results/anonymity_scores/plots/{experiment_id}.png'
    n_bootstrap: int = 1
    plot: bool = False
    normalization: str = 'none'  # 'none' and '0mean' don't affect ks statistic, so just keep it at 'none'. 'whiten' should, but probably gives same result.


    def __post_init__(self):
        self.experiment_id = f'{self.n_bootstrap}:{self.n_patients}:{self.wavelet}:{self.level}:{self.patchsize}:{",".join(self.patch_features)}'
        self.experiment_id = self.experiment_id.replace(':', '-')
        self.save_fp = self.save_fp.format(**self.__dict__)
        self.save_img_fp = self.save_img_fp.format(**self.__dict__)


def parse_args(argv=None) -> Options:
    par = ArgumentParser()
    par.add_arguments(Options, 'args')
    return par.parse_args(argv).args


def get_dset(args, seed):
    # dataset:  chexpert dataset: randomly select all images from only n_patients
    dset = CheXpert_Small(
        use_train_set=True,
        img_transform=tvt.Compose([
            tvt.ToTensor(),
            tvt.CenterCrop((320,320)),
        ]),
        getitem_transform=lambda dct: dct)
    assert dset.labels_csv['Patient'].is_monotonic_increasing, 'error'
    assert dset.labels_csv['Patient'].diff().max() == 1, 'error'
    z = dset.labels_csv['Patient'].unique().copy()
    np.random.default_rng(seed).shuffle(z)
    #  np.random.shuffle(z)
    idxs = dset.labels_csv['index']\
        .loc[dset.labels_csv['Patient'].isin(z[:args.n_patients])].values
    return T.utils.data.Subset(dset, idxs)


def get_model(args):
    deepfix_mdl = DeepFixCompression(
        in_ch=1, in_ch_multiplier=1,
        levels=args.level, wavelet=args.wavelet,
        patch_size=args.patchsize, patch_features=args.patch_features,
        adaptive=0, zero_mean=False,
        how_to_error_if_input_too_small='warn')
    if args.normalization != 'none':
        D = deepfix_mdl.out_shape[-1]
        assert D == 4**args.level * args.patchsize**2 * len(args.patch_features)
        deepfix_mdl = T.nn.Sequential(deepfix_mdl, Normalization(
            D=D, normalization=args.normalization, filepath=(
                f'norms/chexpert_small:{args.wavelet}:{args.level}:{args.patchsize}:{",".join(args.patch_features)}:0.pth')))
    deepfix_mdl.to(args.device)
    return deepfix_mdl


def analyze_dist_matrices(args, cdists:List[T.Tensor], patient_id_matches:List[T.Tensor]):
    # for each bootstrap, analyze the pairwise distance matrix to get:
    # - distribution of distances for images of same patient
    # - distribution of distances for images of different patients
    # - distance between these two distributions
    ks_tests = []
    same_patient = []
    diff_patient = []
    for bootstrap_idx in range(args.n_bootstrap):
        same_ids = patient_id_matches[bootstrap_idx]
        # same patient
        vec1 = cdists[bootstrap_idx][same_ids].cpu().numpy()
        # different patient
        vec2 = cdists[bootstrap_idx][~same_ids].cpu().numpy()
        assert (len(vec1)>0 and len(vec2)>0), 'error related to number of patient_id matches'
        ks_tests.append(scipy.stats.ks_2samp(vec1, vec2))
        same_patient.append(vec1)
        diff_patient.append(vec2)
    #  anderson_test = scipy.stats.anderson_ksamp(same_patient + diff_patient)
    # can't use anderson because samples aren't independent.  they are paired.
    return ks_tests, same_patient, diff_patient


def collator(batch: List[Dict]):
    imgs = T.stack([x.pop('image') for x in batch])
    return imgs, batch


def main():
    args = parse_args()
    print(args)

    deepfix_mdl = get_model(args)
    pdists, patient_id_matches, link_to_original_data = [], [], []
    for bootstrap_idx in range(args.n_bootstrap):
        T.cuda.empty_cache()
        print(f'bootstrap {bootstrap_idx}: get encodings')
        dset = get_dset(args, seed=bootstrap_idx)
        labels = []
        encs = []
        patient_ids = []
        for x, y in T.utils.data.DataLoader(
                dset, batch_size=200, num_workers=5, pin_memory=False,
                collate_fn=collator, shuffle=False):
            x = x.to(args.device, non_blocking=True)
            labels.extend(y)
            patient_ids.extend([dct['labels'].loc['Patient'] for dct in y])
            with T.no_grad():
                encs.append(deepfix_mdl(x))

        print(f'bootstrap {bootstrap_idx}: get pairwise distances')
        link_to_original_data.append(labels)
        assert len(labels) == len(dset)
        encs = T.cat(encs, 0).squeeze(1)
        pdists.append(T.pdist(encs).cpu())
        del encs
        patient_ids = T.tensor(patient_ids, dtype=T.float)  #, device=args.device)
        patient_id_matches.append((T.pdist(patient_ids.reshape(-1,1)) == 0).cpu())
        del patient_ids
        N = len(dset)
        assert patient_id_matches[-1].numel() == pdists[-1].numel() == (N*N-N)/2

    print('analyze dist matrices')

    ks_tests, same_patient, diff_patient = analyze_dist_matrices(
        args, pdists, patient_id_matches)

    #  save cdist and patient_id_matches to a pth file
    makedirs(dirname(args.save_fp), exist_ok=True)
    T.save({
        #  'pdists': pdists, 'patient_id_matches': patient_id_matches,
        #  'link_pdists_to_chexpert_data': link_to_original_data,  #  of form: {'row or col index': metadata}
        #  'distances_same_patient': same_patient,
        #  'distances_diff_patient': diff_patient,
        'ks_tests': ks_tests,
    }, args.save_fp)

    print(f'saved distance matrix to {args.save_fp}')
    kss_mean = np.mean([x.statistic for x in ks_tests])
    kss = [x.statistic for x in ks_tests]
    kss_std = np.std([x.statistic for x in ks_tests])
    ks_pvalue = np.mean([x.pvalue for x in ks_tests])
    print('Averaged KS result', kss_mean, kss)
    for kst in ks_tests:
        print(kst)

    ci = 1.96 * kss_std / (len(ks_tests)**.5)
    print('KS statistic, ci', ci)

    if args.plot:
        ###
        # Plotting below here
        ###

        # how different are the distributions?
        fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2,2, figsize=(10,10))
        _.axis('off')

        vec1, vec2 = np.hstack(same_patient), np.hstack(diff_patient)

        def to_cdf(x):
            # TODO: please check this is right
            vals, edges = np.histogram(x, bins=50)
            vals = (vals - vals.min()) / (vals.max() - vals.min())
            vals /= vals.sum()
            x = np.cumsum(vals)
            return x
        ax1.plot(to_cdf(vec1), label='Same Patient')
        ax1.plot(to_cdf(vec2), label='Different Patient')
        #  print('KS TEST RESULT', ks_result)
        ax1.legend()
        ax1.set_title(f'2-sample KS Test, p={ks_pvalue:.05e}')
        # TODO: consider adding KS test vertical line (like on wikipedia)
        bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if (ks_pvalue < 1e-4):
            ax1.text(0.05, 0.1, transform=ax1.transAxes, bbox=bbox, s=(
                f'KS statistic: {kss_mean:.05f}'
                f'\nSmall p-value ({ks_pvalue:.05e} < 1e-4).'
                '\nEvidence that distributions are different.'))
        else:
            ax1.text(0.05, 0.1, transform=ax1.transAxes, bbox=bbox, s=(
                f'KS statistic: {kss_mean:.05f}'
                f'\nLarge p-value ({ks_pvalue:.05e} > 1e-4).'
                '\nNeed more evidence that distributions are different.'))

        # plot the difference (and show the ks test result somewhere)
        df = pd.DataFrame({'Same Patient': pd.Series(vec1),
                           'Different Patient': pd.Series(vec2)})
        sns.violinplot(
            data=df.melt(value_name='Distance'),
            x='variable', y='Distance', ax=ax2, scale='count', inner='box')
        ax2.set_title('Distribution of pairwise distances')
        ax2.set_xlabel(None)

        ax3.set_title('Pairwise Distances, 1st bootstrap')
        ax3.imshow(scipy.spatial.distance.squareform(pdists[0].cpu().numpy()), vmin=0)

        makedirs(dirname(args.save_img_fp), exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.save_img_fp, bbox_inches='tight')
        print(f'saved plot to {args.save_img_fp}')

        #  plt.show(block=False)
        #  plt.pause(10)
        #  return ks_result


if __name__ == "__main__":
    main()
