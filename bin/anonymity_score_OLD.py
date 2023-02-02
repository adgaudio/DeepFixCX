#!/usr/bin/env python
"""
Get the anonymity score for a deepfixcx encoder model.

python bin/anonymity_score.py --dset chexpert_small:.1:.001 --model waveletmlp:700:1:14:7:32:3:3 --lossfn chexpert_uignore
"""
import shutil
from concurrent import futures as fut
import pickle
import threading
from simple_parsing import ArgumentParser
import dataclasses as dc
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import pandas as pd
import seaborn as sns
from os.path import dirname, exists
from os import makedirs
import torch as T
import torchvision.transforms as tvt
import scipy.stats

from simplepytorch.datasets import CheXpert_Small
from deepfixcx.models import DeepFixCXCompression


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
    wavelet: str = 'coif2'
    patch_features: List[str] = ('l1', )
    device: str = 'cpu'  # cuda doesn't work with multiprocessing
    save_fp: str = './results/anonymity_scores/{experiment_id}.pth'
    save_img_fp: str = './results/anonymity_scores/plots/{experiment_id}.png'
    cache_dir: str = './results/anonymity_scores/cache/{experiment_id}'
    n_bootstrap: int = 1
    plot: bool = False
    parallelization: int = None  # num cpu processes

    def __post_init__(self):
        self.experiment_id = f'{self.n_bootstrap}:{self.n_patients}:{self.wavelet}:{self.level}:{self.patchsize}:{",".join(self.patch_features)}'
        self.experiment_id = self.experiment_id.replace(':', '-')
        self.save_fp = self.save_fp.format(**self.__dict__)
        self.save_img_fp = self.save_img_fp.format(**self.__dict__)
        self.cache_dir = self.cache_dir.format(**self.__dict__)


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
    deepfixcx_mdl = DeepFixCXCompression(
        in_ch=1, in_ch_multiplier=1,
        levels=args.level, wavelet=args.wavelet,
        patch_size=args.patchsize, patch_features=args.patch_features,
        adaptive=0, zero_mean=False,
        how_to_error_if_input_too_small='warn')
    deepfixcx_mdl.to(args.device)
    return deepfixcx_mdl


def get_deepfixcxed_img_and_labels(deepfixcx_model, dset, bootstrap_idx, idx, device):
    dct = dset[idx]
    x = dct['image'].to(device, non_blocking=True)
    patient_id = dct['labels'].loc['Patient']
    x_deepfixcx = deepfixcx_model(x.unsqueeze(0))
    metadata = {'labels': dct['labels'], 'fp': dct['fp'],
                'filesize': x.shape, 'compressed_size': x_deepfixcx.shape}
    return x_deepfixcx, patient_id, metadata


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
        self.locks = set()

    def __call__(self, *args, **kwargs):
        fp = self.get_filepath({k: kwargs[k] for k in self.cache_these_kwargs})
        try:
            if fp in self.locks:
                try:
                    output = T.load(fp, map_location=self.device)['output']
                except:
                    try:
                        output = T.load(fp, map_location=self.device)['output']
                    except:
                        output = T.load(fp, map_location=self.device)['output']
            else:
                raise FileNotFoundError()
            #  print('load from cache')
        except (OSError, KeyError, EOFError, RuntimeError, FileNotFoundError, TimeoutError, pickle.UnpicklingError) as err:
            # pytorch makes for a poor caching system, but it works.
            if fp not in self.locks:
                self.locks.add(fp)
                output = self.wrapped_function(*args, **kwargs)
                try:
                    T.save({'output': output}, fp+'.tmp')
                    shutil.move(fp+'.tmp', fp)
                except:
                    pass  # another process did this.
                #  lock.release()  # do not release the lock to guarantee only runs once
            else:
                print('already computed', err)
            if not isinstance(err, FileNotFoundError):
                print('cache err:', err)
        return output

    def get_filepath(self, cache_kwargs) -> str:
        filename = '_'.join(f'{k}={v}' for k,v in cache_kwargs.items())
        fp = f'{self.cache_dir}/{filename}.pth'
        return fp


def analyze_dist_matrices(args, cdists:List[np.ndarray], patient_id_matches):
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
        vec1 = cdists[bootstrap_idx][same_ids]
        # different patient
        vec2 = cdists[bootstrap_idx][(~same_ids)&T.ones_like(same_ids).triu(1)]
        assert (len(vec1)>0 and len(vec2)>0), 'error related to number of patient_id matches'
        ks_tests.append(scipy.stats.ks_2samp(vec1, vec2))
        same_patient.append(vec1)
        diff_patient.append(vec2)
    #  anderson_test = scipy.stats.anderson_ksamp(same_patient + diff_patient)
    # can't use anderson because samples aren't independent.  they are paired.
    return ks_tests, same_patient, diff_patient

def _pairwise_distv2(bootstrap_idx, idx, args):
    """compute a row of pairwise distances in the pairwise dist matrix"""
    # fetch embeddings from cache (to save compute time) 
    cachefn = CacheToDiskPyTorch(
        get_deepfixcxed_img_and_labels, cache_these_kwargs=['bootstrap_idx', 'idx'],
        cache_dir=args.cache_dir, device=args.device
    )
    deepfixcx_mdl = get_model(args)
    dset = get_dset(args, seed=bootstrap_idx)
    # get first image
    deepfixcxed_img, patient_id, metadata = cachefn(
        deepfixcx_mdl, dset, bootstrap_idx=bootstrap_idx, idx=idx, device=args.device)
    # get distance of other images to this image
    dist_vec = T.zeros(len(dset), device=args.device, dtype=T.float)
    id_matches_vec = T.zeros(len(dset), device=args.device, dtype=T.float)
    for idx2 in range(idx+1, len(dset)):
        another_deepfixcxed_img, another_patient_id, _ = cachefn(
            deepfixcx_mdl, dset, bootstrap_idx=bootstrap_idx, idx=idx2, device=args.device)
        # TODO: is this the best metric?  Earth mover's distance?
        dist = euclidean_dist(deepfixcxed_img, another_deepfixcxed_img)
        id_matches = T.tensor(
            (patient_id == another_patient_id).astype('uint8'), dtype=T.bool)
        dist_vec[idx2] = dist
        id_matches_vec[idx2] = id_matches
    print('...', bootstrap_idx, idx)
    return bootstrap_idx, idx, metadata, dist_vec, id_matches_vec


def main():
    args = parse_args()
    print(args)
    # clear the cache
    print(f'using cache: {args.cache_dir}')
    shutil.rmtree(args.cache_dir, ignore_errors=True)

    # compute pairwise distance matrix
    # Note: use euclidean distance for now.
    print(f'Constructing upper triangular pairwise distance matrices')
    dset_sizes = [len(get_dset(args, i)) for i in range(args.n_bootstrap)]
    cdists = [
        T.zeros((N, N), device=args.device, dtype=T.float) for N in dset_sizes]
    patient_id_matches = [
        # T.same shape as cdist  (N,N) where N is num images for the given n_patients
        T.zeros((N, N), device=args.device, dtype=T.bool) for N in dset_sizes]
    link_to_original_data = {}  # {(bootstrap_idx, cdist_row_idx): patient_metadata}  to identify the original input image if desired in future

    with fut.ProcessPoolExecutor(max_workers=args.parallelization) as executor:
    #  with fut.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for bootstrap_idx in range(args.n_bootstrap):
            print('bootstrap_idx', bootstrap_idx)
            print('queue jobs')
            N = dset_sizes[bootstrap_idx]
            futures.extend(executor.map(
                _pairwise_distv2, [bootstrap_idx]*N, range(N), [args]*N))
            # assemble rows of the pairwise distance matrix
    print('collect results')
    for future in futures:
        bootstrap_idx, idx, metadata, dist, id_matches = future#.result()
        cdists[bootstrap_idx][idx, :] = dist
        patient_id_matches[bootstrap_idx][idx, :] = id_matches
        link_to_original_data[(bootstrap_idx, idx)] = metadata


            #  for idx in range(len(dset)):
            #      #  sys.stdout.write(f'{idx}.')
            #      #  sys.stdout.flush()
            #      deepfixcxed_img, patient_id, metadata = cached__get_deepfixcxed_img_and_labels(
            #          deepfixcx_mdl, dset, bootstrap_idx=bootstrap_idx, idx=idx, device=args.device)
            #      link_to_original_data[(bootstrap_idx, idx)] = metadata

            #      for idx2 in range(idx+1, len(dset)):
            #          executor.submit(
            #              _pairwise_dist, bootstrap_idx, idx, deepfixcxed_img, patient_id, idx2)
    print('')  # newline for the stdout.write(...)
    print('analyze dist matrices')

    cdists = [x.to('cpu', non_blocking=True).numpy() for x in cdists]
    #  M = cdist.triu()
    ks_tests, same_patient, diff_patient = analyze_dist_matrices(
        args, cdists, patient_id_matches)

    #  save cdist and patient_id_matches to a pth file
    makedirs(dirname(args.save_fp), exist_ok=True)
    T.save({
        'cdists': cdists, 'patient_id_matches': patient_id_matches,
        'link_cdist_to_chexpert_data': link_to_original_data,  #  of form: {'row or col index': metadata}
        'distances_same_patient': same_patient,
        'distances_diff_patient': diff_patient,
        'ks_tests': ks_tests,
    }, args.save_fp)
    print(f'saved distance matrix to {args.save_fp}')
    ks_pvalue = np.mean([x.pvalue for x in ks_tests])
    kss_mean = np.mean([x.statistic for x in ks_tests])
    kss_std = np.std([x.pvalue for x in ks_tests])
    print('Averaged KS result', kss_mean, ks_pvalue)
    print([x.statistic for x in ks_tests])

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
        ax3.imshow(cdists[0], vmin=0)

        makedirs(dirname(args.save_img_fp), exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.save_img_fp, bbox_inches='tight')
        print(f'saved plot to {args.save_img_fp}')

        #  plt.show(block=False)
        #  plt.pause(10)
        #  return ks_result


if __name__ == "__main__":
    main()
