"""
Get the anonymity score for a deepfix encoder model.

python bin/anonymity_score.py --dset chexpert_small:.1:.001 --model waveletmlp:700:1:14:7:32:3:3 --lossfn chexpert_uignore
"""
from concurrent.futures import ThreadPoolExecutor
from simple_parsing import ArgumentParser
import dataclasses as dc
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import pandas as pd
import seaborn as sns
import sys
from os.path import dirname
from os import makedirs
import torch as T
import torchvision.transforms as tvt
import scipy.stats

from simplepytorch.datasets import CheXpert_Small
from deepfix import train
from deepfix.models import DeepFixCompression


def euclidean_dist(vec1, vec2):
    """Compute B Distances between two tensors of shape (B, ...), returning a pairwise distance matrix"""
    B = vec1.shape[0]
    assert vec1.shape[0] == vec2.shape[0]
    return ((vec1.reshape(B, 1, -1) - vec2.reshape(1, B, -1))**2).sum((2)).sqrt()


@dc.dataclass
class Options:
    n_patients:int = 5
    level:int = 5
    patchsize:int = 64
    wavelet:str = 'coif1'
    patch_features:List[str] = 'l1'
    device:str = 'cuda'
    save_fp:str = './results/anonymity_scores/{experiment_id}.pth'
    save_img_fp:str = './results/anonymity_scores/plots/{experiment_id}.png'
    cache_dir:str = './results/anonymity_scores/cache/{experiment_id}'

    def __post_init__(self):
        self.experiment_id = f'{self.n_patients}_{self.wavelet}:{self.level}:{self.patchsize}:{",".join(self.patch_features)}'
        self.save_fp = self.save_fp.format(**self.__dict__)
        self.save_img_fp = self.save_img_fp.format(**self.__dict__)
        self.cache_dir = self.cache_dir.format(**self.__dict__)


def parse_args(argv=None) -> Options:
    par = ArgumentParser()
    par.add_arguments(Options, 'args')
    return par.parse_args(argv).args


def get_model_and_dset(args):
    # model
    deepfix_mdl = DeepFixCompression(
        in_ch=1, in_ch_multiplier=1,
        levels=args.level, wavelet=args.wavelet,
        patch_size=args.patchsize, patch_features=args.patch_features,
        how_to_error_if_input_too_small='warn')
    deepfix_mdl.to(args.device)
    # dataset:  chexpert dataset: randomly select all images from only n_patients
    dset = CheXpert_Small(
        use_train_set=True,
        img_transform=tvt.Compose([
            tvt.RandomCrop((320, 320)),
            tvt.ToTensor(),  # full res 1024x1024 imgs
        ]),
        getitem_transform=lambda dct: dct)
    z = dset.labels_csv['Patient'].unique().copy()
    np.random.shuffle(z)
    idxs = dset.labels_csv['index']\
        .loc[dset.labels_csv['Patient'].isin(z[:args.n_patients])].values
    dset = T.utils.data.Subset(dset, idxs)
    return deepfix_mdl, dset


def get_deepfixed_img_and_labels(deepfix_model, dset, idx, device):
    dct = dset[idx]
    x = dct['image'].to(device, non_blocking=True)
    patient_id = dct['labels'].loc['Patient']
    x_deepfix = deepfix_model(x.unsqueeze(0))
    metadata = {'labels': dct['labels'], 'fp': dct['fp'],
                'filesize': x.shape, 'compressed_size': x_deepfix.shape}
    return x_deepfix, patient_id, metadata


class CacheToDiskPyTorch:
    """Decorator to cache function calls on disk.

    Only do cache lookups / saves based on specified keyword arguments and the
    given directory.  The keyword arguments used for cacheing must have values
    that can be represented meaningfully as a string (like they should be str or int).
    Use Pytorch functions T.load(...) and T.save(...) to read/write.
    """
    def __init__(self, function, cache_these_kwargs:List[str], cache_dir:str, device:str):
        self.cache_dir = cache_dir
        makedirs(self.cache_dir, exist_ok=True)
        self.wrapped_function = function
        self.device = device
        self.cache_these_kwargs = cache_these_kwargs

    def __call__(self, *args, **kwargs):
        fp = self.get_filepath({k: kwargs[k] for k in self.cache_these_kwargs})
        try:
            output = T.load(fp, map_location=self.device)['output']
            #  print('load from cache')
        except FileNotFoundError:
            output = self.wrapped_function(*args, **kwargs)
            T.save({'output': output}, fp)
            #  print('       SAVE to cache')
        return output

    def get_filepath(self, cache_kwargs) -> str:
        filename = '_'.join(f'{k}={v}' for k,v in cache_kwargs.items())
        fp = f'{self.cache_dir}/{filename}.pth'
        return fp


def main():
    args = parse_args()
    deepfix_mdl, dset = get_model_and_dset(args)
    # compute pairwise distance matrix
    # Note: use euclidean distance for now.
    N = len(dset)
    print(f'Constructing {N}x{N} upper triangular pairwise distance matrix')
    cdist = T.zeros((N, N), device=args.device, dtype=T.float)  # T.Tensor
    patient_id_matches = T.zeros((N, N), dtype=T.bool)  # T.same shape as cdist  (N,N) where N is num patients
    link_to_original_data = {}  # to identify the original input image if desired in future

    cached__get_deepfixed_img_and_labels = CacheToDiskPyTorch(
        get_deepfixed_img_and_labels, cache_these_kwargs=['idx'],
        cache_dir=args.cache_dir, device=args.device
    )
    print(f'using cache: {args.cache_dir}')

    started = set()
    def _pairwise_dist(idx, deepfixed_img, patient_id, idx2):
        another_deepfixed_img, another_patient_id, _ = cached__get_deepfixed_img_and_labels(
            deepfix_mdl, dset, idx=idx2, device=args.device)
        # TODO: is this the best metric?  Earth mover's distance?
        cdist[idx, idx2] = euclidean_dist(
            deepfixed_img, another_deepfixed_img)
        patient_id_matches[idx, idx2] = T.tensor(
            (patient_id == another_patient_id).astype('uint8'), dtype=T.bool)
    def _pairwise_distv2(idx, idx2):
        if idx not in started:
            started.add(idx)
            sys.stdout.write(f'{idx}.')
            sys.stdout.flush()
        deepfixed_img, patient_id, metadata = cached__get_deepfixed_img_and_labels(
            deepfix_mdl, dset, idx=idx, device=args.device)
        link_to_original_data[idx] = metadata
        _pairwise_dist(idx, deepfixed_img, patient_id, idx2)
    with ThreadPoolExecutor(max_workers=20) as executor:
        #  for idx in range(N):
            #  for idx2 in range(idx+1, N):
                #  executor.submit(_pairwise_distv2, idx, idx2)
        for idx in range(N):
            sys.stdout.write(f'{idx}.')
            sys.stdout.flush()
            deepfixed_img, patient_id, metadata = cached__get_deepfixed_img_and_labels(
                deepfix_mdl, dset, idx=idx, device=args.device)
            link_to_original_data[idx] = metadata

            for idx2 in range(idx+1, N):
                executor.submit(
                    _pairwise_dist, idx, deepfixed_img, patient_id, idx2)
    print()  # newline for the stdout.write(...)

    cdist = cdist.to('cpu')
    #  M = cdist.triu()

    vec1 = distances_same_patient = cdist[patient_id_matches].numpy()
    vec2 = distances_diff_patient = cdist[(~patient_id_matches)&T.ones_like(patient_id_matches).triu(1)].numpy()
    assert (len(vec1)>0 and len(vec2)>0), 'error related to number of patient_id matches'

    ks_result = scipy.stats.ks_2samp(vec1, vec2)

    #  save cdist and patient_id_matches to a pth file
    makedirs(dirname(args.save_fp), exist_ok=True)
    T.save({
        'cdist': cdist, 'patient_id_matches': patient_id_matches,
        'link_cdist_to_chexpert_data': link_to_original_data,  #  of form: {'row or col index': metadata}
        'distances_same_patient': distances_same_patient,
        'distances_diff_patient': distances_diff_patient,
        'ks_test': ks_result,
    }, args.save_fp)
    print(f'saved distance matrix to {args.save_fp}')

    # how different are the distributions?
    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2,2, figsize=(10,10))
    _.axis('off')

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
    ax1.set_title(f'2-sample KS Test, p={ks_result.pvalue:.05e}')
    # TODO: consider adding KS test vertical line (like on wikipedia)
    bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if (ks_result.pvalue < 1e-4):
        ax1.text(0.05, 0.1, transform=ax1.transAxes, bbox=bbox, s=(
            f'KS statistic: {ks_result.statistic:.05f}'
            f'\nSmall p-value ({ks_result.pvalue:.05e} < 1e-4).'
            '\nEvidence that distributions are different.'))
    else:
        ax1.text(0.05, 0.1, transform=ax1.transAxes, bbox=bbox, s=(
            f'KS statistic: {ks_result.statistic:.05f}'
            f'\nLarge p-value ({ks_result.pvalue:.05e} > 1e-4).'
            '\nNeed more evidence that distributions are different.'))

    # plot the difference (and show the ks test result somewhere)
    df = pd.DataFrame({'Same Patient': pd.Series(vec1),
                       'Different Patient': pd.Series(vec2)})
    sns.violinplot(
        data=df.melt(value_name='Distance'),
        x='variable', y='Distance', ax=ax2, scale='count', inner='box')
    ax2.set_title('Distribution of pairwise distances')
    ax2.set_xlabel(None)

    ax3.set_title('Pairwise Distances')
    ax3.imshow(cdist.numpy(), vmin=0)

    makedirs(dirname(args.save_img_fp), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.save_img_fp, bbox_inches='tight')
    print(f'saved plot to {args.save_img_fp}')

    #  plt.show(block=False)
    #  plt.pause(10)


if __name__ == "__main__":
    main()
