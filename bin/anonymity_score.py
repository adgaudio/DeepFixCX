"""
Get the anonymity score for a deepfix encoder model.

python bin/anonymity_score.py --dset chexpert_small:.1:.001 --model waveletmlp:700:1:14:7:32:3:3 --lossfn chexpert_uignore
"""
from simple_parsing import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import torch as T
import torchvision.transforms as tvt
import scipy.stats

from simplepytorch.datasets import CheXpert_Small
from deepfix import train


def euclidean_dist(vec1, vec2):
    """Compute B Distances between two tensors of shape (B, ...), returning a pairwise distance matrix"""
    B = vec1.shape[0]
    assert vec1.shape[0] == vec2.shape[0]
    return ((vec1.reshape(B, 1, -1) - vec2.reshape(1, B, -1))**2).sum((2)).sqrt()


def get_model_and_dset(n_patients:int):
    # get default model we used to train the mlp
    # note: we only care about the dataset and compression model.
    p = ArgumentParser()
    p.add_arguments(train.TrainOptions, dest='TrainOptions')
    args = p.parse_args().TrainOptions
    print(args)
    #  assert args.dset.startswith('chexpert') and args.dset.endswith(':index,Patient'), 'please pass something like --dset chexpert_small:.1:.01:index,Patient'
    cfg = train.train_config(args)
    deepfix_mdl = cfg.model.compression_mdl.eval()
    # chexpert dataset: randomly select all images from only n_patients
    dset = CheXpert_Small(
        use_train_set=True,
        img_transform=tvt.Compose([
            #  tvt.RandomCrop((320, 320)),
            tvt.ToTensor(),  # full res 1024x1024 imgs
        ]),
        getitem_transform=lambda dct: dct)

    z = dset.labels_csv['Patient'].unique().copy()
    np.random.shuffle(z)
    idxs = dset.labels_csv['index'].loc[dset.labels_csv['Patient'].isin(z[:n_patients])].values
    dset = T.utils.data.Subset(dset, idxs)
    #  dset = cfg.val_dset
    return deepfix_mdl, dset, cfg.experiment_id, cfg.device


def get_deepfixed_img_and_labels(idx):
    dct = dset[idx]
    x = dct['image'].to(device, non_blocking=True)
    patient_id = dct['labels'].loc['Patient']
    x_deepfix = deepfix_mdl(x.unsqueeze(0))
    metadata = {'labels': dct['labels'], 'fp': dct['fp'],
                'filesize': x.shape, 'compressed_size': x_deepfix.shape}
    return x_deepfix, patient_id, metadata


if __name__ == "__main__":
    n_patients = 50  # 200  # TODO: make this large enough after done debugging until the results are stable across multiple runs
    deepfix_mdl, dset, experiment_id, device = get_model_and_dset(n_patients)

    # compute pairwise distance matrix
    # Note: use euclidean distance for now.
    N = len(dset)
    print(f'Constructing {N}x{N} upper triangular pairwise distance matrix')
    cdist = T.zeros((N, N), device=device, dtype=T.float)  # T.Tensor
    patient_id_matches = T.zeros((N, N), dtype=T.bool)  # T.same shape as cdist  (N,N) where N is num patients
    link_to_original_data = []  # to identify the original input image if desired in future

    # TODO: wrap get_deepfixed_img_and_labels(x) with a cache access layer (to get massive speedup)
    # TODO: whatever you conda install to get this working, make note in INSTALL.md.
    # * Make sure it caches to HARD DISK, not memory!!!  A standard RAM cache probably won't work very well
    # ** You might want to make a separate file for each x,y (but maybe try not to have more than, say, 5k files in one directory)
    # for example: https://pypi.org/project/cache-to-disk/
    # for example: https://tohyongcheng.github.io/python/2016/06/07/persisting-a-cache-in-python-to-disk.html
    for idx in range(N):
        sys.stdout.write(f'{idx}.')
        sys.stdout.flush()
        deepfixed_img, patient_id, metadata = get_deepfixed_img_and_labels(idx)
        link_to_original_data.append(metadata)
        for idx2 in range(idx+1, N):
            another_deepfixed_img, another_patient_id, _ = get_deepfixed_img_and_labels(idx2)
            # TODO: is this the best metric?  Earth mover's distance?
            cdist[idx, idx2] = euclidean_dist(
                deepfixed_img, another_deepfixed_img)
            patient_id_matches[idx, idx2] = T.tensor(
                (patient_id == another_patient_id).astype('uint8'), dtype=T.bool)
    print()  # newline for the stdout.write(...)

    cdist = cdist.to('cpu')
    #  M = cdist.triu()

    #  save cdist and patient_id_matches to a pth file
    save_fp = f'./results/{experiment_id}/anonymity_score_matrix.pth'
    T.save({
        'cdist': cdist, 'patient_id_matches': patient_id_matches,
        'link_cdist_to_chexpert_data': link_to_original_data}, save_fp)
    print(f'save distance matrix to {save_fp}')

    vec1 = cdist[patient_id_matches].numpy()
    vec2 = cdist[(~patient_id_matches)&T.ones_like(patient_id_matches).triu(1)].numpy()
    assert (len(vec1)>0 and len(vec2)>0), 'error related to number of patient_id matches'

    # how different are the distributions?
    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2,2, figsize=(12,12))
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
    ks_result = scipy.stats.ks_2samp(vec1, vec2)
    #  print('KS TEST RESULT', ks_result)
    ax1.legend()
    ax1.set_title(f'2-sample KS Test, p={ks_result.pvalue:.05e}')
    # TODO: consider adding KS test vertical line (like on wikipedia)
    # TODO: consider saying "Statistically Different?  yes|not enough evidence" with ax.text, (only yes if p-value < 1e-4)
    """
    """
    if (ks_result.pvalue < 1e-4):
        ax1.text(20, 0.4, s=f'Very Strong Evidence.')
    else:
        ax1.text(20, 0.4, s=f'Not Enough Evidence.')


    # plot the difference (and show the ks test result somewhere)
    df = pd.DataFrame({'Same Patient': pd.Series(vec1),
                       'Different Patient': pd.Series(vec2)})
    sns.violinplot(data=df.melt(value_name='Distance'), x='variable', y='Distance', ax=ax2)
    ax2.set_title('Distribution of pairwise distances')
    ax2.set_xlabel(None)

    ax3.set_title('Pairwise Distances')
    ax3.imshow(cdist.numpy(), vmin=0)

    save_fp = f'./results/{experiment_id}/anonymity_score_plot.png'
    print(f'save plot to {save_fp}')
    fig.tight_layout()
    fig.savefig(save_fp, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(10)
