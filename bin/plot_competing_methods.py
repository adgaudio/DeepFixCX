from waveletfix.dsets import get_dset_chexpert
from waveletfix.models import WaveletFixImg2Img, MedianPool2d
import re
from skimage.metrics import structural_similarity as ssim
import torch as T
from torchvision.transforms import GaussianBlur, Resize
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from subprocess import check_output
import cv2


def plot_accuracy_DNNs():
    # get below results from the below shell commands
    #    $ chexpert=true ./bin/table_best_accuracy.sh 4.C8
    #    $ chexpert=true ./bin/table_best_accuracy.sh 2.C28
    #    $ chexpert=true ./bin/table_best_accuracy.sh 2.C29
    #    $ chexpert=true ./bin/table_best_accuracy.sh 2.C30
    # results are the 'test_ROC_AUC AVG' of the model (across J,P and epochs) with highest 'val_ROC_AUC AVG' accuracy.
    chexpert_baselines = pd.DataFrame([
        # below:  1st line:  baseline.  2nd line:  model with highest val roc auc avg.  3rd line (commented out): of models with highest val roc auc avg, choose that with highest test roc auc avg.  Third line looks best but I think it isn't correct way to do model selection.
        ("DenseNet121 vs $WaveletFix$", 'DNN', 0.868),  # 4.C8
        ("DenseNet121 vs $WaveletFix$",
         '$WaveletFix$ (J=1 P=115) -> DNN', 0.876),  # 2.C28
        # ("DenseNet121 vs $WaveletFix$", '$WaveletFix$ (J=1 P=160) -> DNN', 0.881),  # 2.C28

        ("EfficientNet-b0 vs $WaveletFix$", 'DNN', 0.874),  # 4.C8
        ("EfficientNet-b0 vs $WaveletFix$",
         '$WaveletFix$ (J=1 P=160) -> DNN', 0.871),  # 2.C30
        # ("EfficientNet-b0 vs $WaveletFix$", '$WaveletFix$ (J=1 P=115) -> DNN', 0.877),  # 2.C30

        ("EfficientNetV2_m vs $WaveletFix$", 'DNN', .869),  # 4.C8
        ("EfficientNetV2_m vs $WaveletFix$",
         '$WaveletFix$ (J=1 P=115) -> DNN', .869),  # 2.C33

        ('MDMLP_320 vs $WaveletFix$', 'DNN', .831),
        ('MDMLP_320 vs $WaveletFix$', '$WaveletFix$ (J=1 P=79) -> DNN', .848),  # 2.C35
        # ('MDMLP_320 vs $WaveletFix$', '$WaveletFix$ (J=1 P=37) -> DNN', .854),  # 2.C35

        ('VOLO_d1_224 vs $WaveletFix$', 'DNN', .805),
        ('VOLO_d1_224 vs $WaveletFix$', '$WaveletFix$ (J=1 P=115) -> DNN', .837),  # 2.C29
        # ('VOLO_d1_224 vs $WaveletFix$', '$WaveletFix$ (J=2 P=79) -> DNN', .843),  # 2.C29

        #
        ('CoAtNet_224 vs $WaveletFix$', 'DNN', .828),
        ('CoAtNet_224 vs $WaveletFix$', '$WaveletFix$ (J=1 P=160) -> DNN', .831),  # 2.C36
        # ('CoAtNet_224 vs $WaveletFix$', '$WaveletFix$ (J=1 P=37) -> DNN', .843),  # 2.C36

        ('VIP_s7 vs $WaveletFix$', 'DNN', .801),  # 4.C8
        ('VIP_s7 vs $WaveletFix$', '$WaveletFix$ (J=1 P=115) -> DNN', .801),  # 2.C33
        # not evaluated on all models.

        # ('ResNet18 vs $WaveletFix$', 'DNN', 0.864),  # 4.C8
        # ('ResNet18 vs $WaveletFix$', '$WaveletFix$ (J=1 P=115) -> DNN', .845),  # 2.C33

    ], columns=['Experiment', 'Method', 'Prediction Performance (Test Set AVG ROC AUC)'])

    # chexpert, accuracy with DNNs
    # ... the table is easier to read than the figure
    # fig, ax = plt.subplots(figsize=(7,4))
    # fig.subplots_adjust(bottom=.20)
    # barplot = sns.barplot(
    #     data=chexpert_baselines,
    #     x='Experiment', y='Prediction Performance (Test Set AVG ROC AUC)',
    #     hue='Method', ax=ax)
    # barplot.set_ylim(.8, .9)
    # barplot.set_title("$WaveletFix$ Improves Prediction Performance of DNNs")
    # barplot.set_xticklabels(
    #     barplot.get_xticklabels(), rotation=10, horizontalalignment='center')
    # barplot.legend(loc='lower right')
    # savefp = 'results/plots/accuracy_DNNs_vs_WaveletFix.png'
    # barplot.figure.savefig(savefp, bbox_inches='tight')
    chexpert_baselines.to_latex(savefp.replace('.png', '.tex'))
    print('save to:', savefp.replace('.png', '.tex'))

    return fig, chexpert_baselines


# chexpert, accuracy with median and blur filters

# idea: for each blur, find closest waveletfix.  get perf on that waveletfix. compare


def blur(tensor_img: T.Tensor, kernel_size: int) -> T.Tensor:
    f = GaussianBlur(kernel_size + 1 - kernel_size % 2, float(kernel_size))
    g = T.nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
    return g(f(tensor_img))


def median(tensor_img: T.Tensor, kernel_size: int, min_size=(64, 64)):
    f = MedianPool2d(kernel_size=kernel_size,
                     stride=kernel_size, padding=0, min_size=min_size)
    return f(tensor_img)


def find_waveletfix_model_most_comparable_to(competing_method, kernel_sizes):
    dset = get_dset_chexpert(.9, .1, small=True, labels='diagnostic', epoch_size=15000)[
        0]['train_dset']
    fig, axs = plt.subplots(2, len(kernel_sizes), figsize=(
        10, 10/len(kernel_sizes)+1.6/8*10))
    for n, kernel_size in enumerate(kernel_sizes):
        img = dset[0][0]
        blurred = competing_method(img.unsqueeze(0), kernel_size)
        score, J, P, waveletfix_img = _find_closest_waveletfix(
            img, blurred, kernel_size)
        # print(kernel_size, ':', J, P, ':', '1/err', score)
        # fig, (c, a,b) = plt.subplots(1,3, figsize=(8,2.75))
        # c.imshow(img.squeeze().numpy())
        # c.set_title('Original Image')
        # c.axis('off')
        a = axs[0, n]
        a.imshow(blurred.squeeze().numpy())
        a.set_title(f'K={kernel_size}')
        b = axs[1, n]
        b.imshow(waveletfix_img.squeeze().numpy())
        b.set_title(f'J={J} P={P}')
        [ax.axis('off') for ax in [a, b]]
    for ax in [axs[0, 0], axs[1, 0]]:
        ax.axis('on')
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
    axs[0, 0].set_ylabel(f'{competing_method.__name__.capitalize()}')
    axs[1, 0].set_ylabel(f'$WaveletFix$')
    savefp = f'results/plots/img_{competing_method.__name__}_vs_WaveletFix.png'
    fig.tight_layout()
    fig.savefig(savefp, bbox_inches='tight')
    print('save to:', savefp)


def _find_closest_waveletfix(img: T.Tensor, blurred_img, K):
    H = img.shape[-2]
    best = [0, 0, 0, None]
    blurred_img = Resize(img.shape[-2:])(blurred_img).squeeze(0).numpy()
    for J in range(1, int(np.log2(H))):
        for P in range(1, 161):
            if P <= H / 2**J:
                waveletfix_img = WaveletFixImg2Img(
                    1, J, P, restore_orig_size=True)(img.unsqueeze(0))
                z = 1 / \
                    np.sqrt(
                        ((blurred_img - waveletfix_img.squeeze(0).squeeze(0).numpy())**2).sum())
                # print(J, P,z)
                # print('...', z)
                if z > best[0]:
                    best[:] = [z, J, P, waveletfix_img]
    # print(best[:3])
    return best


def plot_accuracy_blur_median(plot=True):
    df1 = pd.DataFrame([
        ('K=4 vs\nP=80', 'Blur', 'Blur, K={K}',  .856), (
            'K=4 vs\nP=80', 'Blur', r'closest $WaveletFix$, J=1 P={P}', .883),
        ('K=8 vs\nP=40', 'Blur', 'Blur, K={K}',  .854), (
            'K=8 vs\nP=40', 'Blur', r'closest $WaveletFix$, J=1 P={P}', .855),
        ('K=16 vs\nP=20', 'Blur', 'Blur, K={K}', .792), (
            'K=16 vs\nP=20', 'Blur', r'closest $WaveletFix$, J=1 P={P}', .843),
        ('K=32 vs\nP=10', 'Blur', 'Blur, K={K}', .757), (
            'K=32 vs\nP=10', 'Blur', r'closest $WaveletFix$, J=1 P={P}', .826),
        ('K=64 vs\nP=5', 'Blur', 'Blur, K={K}', .741),  (
            'K=64 vs\nP=5', 'Blur', r'closest $WaveletFix$, J=1 P={P}', .785),
        # \hline
        ('K=3 vs\nP=80', 'Median', 'Median, K={K}',  .859), (
            'K=3 vs\nP=80', 'Median', r'closest $WaveletFix$, J=1 P={P}',  .883),
        ('K=5 vs\nP=80', 'Median', 'Median, K={K}',  .854), (
            'K=5 vs\nP=80', 'Median', r'closest $WaveletFix$, J=1 P={P}',  .883),
        ('K=9  vs\nP=40', 'Median', 'Median, K={K}', .858), (
            'K=9  vs\nP=40', 'Median', r'closest $WaveletFix$, J=1 P={P}', .855),
        ('K=15 vs\nP=40', 'Median', 'Median, K={K}', .850), (
            'K=15 vs\nP=40', 'Median', r'closest $WaveletFix$, J=1 P={P}', .855),
        ('K=25 vs\nP=18', 'Median', 'Median, K={K}', .831), (
            'K=25 vs\nP=18', 'Median', r'closest $WaveletFix$, J=1 P={P}', .840),
        ('K=50 vs\nP=6', 'Median', 'Median, K={K}', .800),  (
            'K=50 vs\nP=6', 'Median', r'closest $WaveletFix$, J=1 P={P}', .791),
        ('K=75 vs\nP=4', 'Median', 'Median, K={K}', .761),  (
            'K=75 vs\nP=4', 'Median', r'closest $WaveletFix$, J=1 P={P}', .766),
    ], columns=[
        'Experiment', 'Competing Method', 'Method', 'Prediction Perf (Test Set AVG ROC AUC)',
    ])
    if plot:
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
        fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4))
        for competing_method, ax, fig in zip(['Median', 'Blur'], [ax1, ax2], [fig1, fig2]):
            ax.set_title(f'Comparison to: {competing_method}()')
            dfz = df1[df1['Competing Method'] == competing_method]
            sns.barplot(
                data=dfz,
                x='Experiment', y='Prediction Perf (Test Set AVG ROC AUC)',
                hue='Method', ax=ax,
            )
            ax.hlines(
                0.874, -.5, dfz.shape[0]/2, label='DNN (EfficientNet-b0, no privatized compression)', linestyle='dashed', color='gray')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=10,
                               horizontalalignment='center')
            ax.set_ylim(.7, .9)
            ax.legend(loc='lower right')
            fig.subplots_adjust(bottom=.2)
            savefp = f'results/plots/accuracy_{competing_method}_vs_WaveletFix.png'
            print('save to: ', savefp)
            fig.savefig(savefp, bbox_inches='tight')
    return df1


def plot_ssim_blur_median():
    experiments = [
        # experiments_blur = [
        {'Experiment': 'K=4 vs\n J=1 P=80', 'Type': 'Blur', 'Method': 'Blur, K={K}'},
        {'Experiment': 'K=4 vs\n J=1 P=80', 'Type': 'Blur',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=8 vs\n J=1 P=40', 'Type': 'Blur', 'Method': 'Blur, K={K}'},
        {'Experiment': 'K=8 vs\n J=1 P=40', 'Type': 'Blur',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=16 vs\n J=1 P=20',
            'Type': 'Blur', 'Method': 'Blur, K={K}'},
        {'Experiment': 'K=16 vs\n J=1 P=20', 'Type': 'Blur',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=32 vs\n J=1 P=10',
            'Type': 'Blur', 'Method': 'Blur, K={K}'},
        {'Experiment': 'K=32 vs\n J=1 P=10', 'Type': 'Blur',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=64 vs\n J=1 P=5', 'Type': 'Blur', 'Method': 'Blur, K={K}'},
        {'Experiment': 'K=64 vs\n J=1 P=5', 'Type': 'Blur',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        # ]
        # experiments_median = [
        {'Experiment': 'K=3 vs\n J=1 P=80',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=3 vs\n J=1 P=80', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=5 vs\n J=1 P=80',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=5 vs\n J=1 P=80', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=9 vs\n J=1 P=40',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=9 vs\n J=1 P=40', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=15 vs\n J=1 P=40',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=15 vs\n J=1 P=40', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=25 vs\n J=1 P=18',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=25 vs\n J=1 P=18', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=50 vs\n J=1 P=6',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=50 vs\n J=1 P=6', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
        {'Experiment': 'K=75 vs\n J=1 P=4',
            'Type': 'Median', 'Method': 'Median, K={K}'},
        {'Experiment': 'K=75 vs\n J=1 P=4', 'Type': 'Median',
            'Method': 'closest $WaveletFix$, J=1 P={P}'},
    ]
    loader = []

    dset = get_dset_chexpert(.9, .1, small=True,
                             labels='diagnostic', epoch_size=15000)[0]
    loader = dset['test_loader']

    results = {}
    for orig_img, _ in loader:
        orig_img = orig_img.detach()
        batch_size = orig_img.shape[0]
        resize_fn = T.nn.Upsample(orig_img.shape[-2:])
        for dct in experiments:
            if 'Median' in dct['Method']:
                K, = re.search(r'K=(\d+)', dct['Experiment']).groups()
                new_img = resize_fn(median(orig_img, int(K)))
            elif 'Blur' in dct['Method']:
                K, = re.search(r'K=(\d+)', dct['Experiment']).groups()
                new_img = resize_fn(blur(orig_img, int(K)))
            elif 'WaveletFix' in dct['Method']:
                J, P = re.search(r'J=(\d+) P=(\d+)',
                                 dct['Experiment']).groups()
                _dfx = WaveletFixImg2Img(
                    1, int(J), int(P), restore_orig_size=True)
                new_img = _dfx(orig_img)
            else:
                raise NotImplementedError()
            dct.setdefault('SSIM', [])
            dct.setdefault('MSE', [])
            dct['SSIM'].extend([
                ssim(a.numpy(), b.numpy(), win_size=3) for a, b in zip(
                    new_img.permute(0, 2, 3, 1).squeeze().unbind(0),
                    orig_img.permute(0, 2, 3, 1).squeeze().unbind(0))])
            dct['MSE'].extend(
                np.sqrt(((new_img - orig_img)**2)
                        .reshape(batch_size, -1).sum(1)).numpy())

    df = pd.DataFrame(experiments).set_index(
        ['Type', 'Experiment', 'Method']).applymap(np.mean).reset_index()
    df = df.set_index('Type')

    # plots
    for method in ['Blur', 'Median']:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.subplots_adjust(bottom=.20)
        bp = sns.barplot(
            data=df.loc[method], x='Experiment', y='SSIM', hue='Method')
        # rotate the labels
        bp.set_xticklabels(
            bp.get_xticklabels(), rotation=10, horizontalalignment='center')
        ax.legend(loc='lower right')
        ax.set_ylim(0.5, .85)
        # save figure and csv data to file
        savefp = f'results/plots/ssim_{method}_vs_WaveletFix.png'
        print('save to:', savefp)
        fig.savefig(
            savefp, bbox_inches='tight')
        df.loc[method].to_csv(savefp.replace('.png', '.csv'))
    return df

    # idea: compare accuracy differences.
    # compare compression on disk differences.
    # compare ssim to original differences.
    # compare speed differences.
    # grouped bar plot.

    # dashed gray hline for efficientnet-b0


def _get_odr_imr(orig_imgs: T.Tensor, new_imgs: T.Tensor):
    """
    Compute the ODR (On-Disk Ratio) and IMR (In-Memory Ratio) between
    `orig_imgs[i]` and its corresponding `new_imgs[i]` for all `i`.

    orig_imgs and new_imgs both have shape (B,C,H,W) and (B,C,H',W') where B is
    batch size and C is channels.  Each tensor has B images.

    Returns (IMR, list[ODR])
    """
    imr = 100 * np.prod(new_imgs.shape[1:]) / np.prod(orig_imgs.shape[1:])
    odrs = []
    for orig_im, new_im in zip(orig_imgs.unbind(0), new_imgs.unbind(0)):
        # save to disk.
        # get filesize.
        sizes = []
        for fp, im in [('/tmp/orig_img.jpg', orig_im),
                       ('/tmp/new_img.jpg', new_im)]:
            cv2.imwrite(
                fp,
                (im.permute(1, 2, 0).squeeze().cpu().numpy()
                 * 255).round().astype('uint8'),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
            sizes.append(float(re.match('(\d+)', check_output(
                f'du -sb {fp}', shell=True).decode()).group(1)))
        odrs.append(sizes[1] / sizes[0])
    return imr, odrs


def plot_compression_blur_median():
    dset = get_dset_chexpert(.9, .1, small=True,
                             labels='diagnostic', epoch_size=15000)[0]
    loader = dset['test_loader']

    data_imrs = {}
    # ('Median', K): Num
    # ('Blur', K): Num
    # ('$WaveletFix$', P): Num
    data_odrs = {
        # ('Median', K): [Num, ...]
        # ('Blur', K): [Num, ...]
        # ('$WaveletFix$', P): [Num, ...]
    }
    img_size = {}
    closest_waveletfix_for_blur = [80, 40, 20, 10, 5]
    closest_waveletfix_for_median = [80, 80, 40, 40, 18, 6, 4]
    for orig_img, _ in loader:
        orig_img = orig_img.detach()
        instructions = [
            ('Blur', blur, [4, 8, 16, 32, 64]),
            ('Median', (lambda img, K: median(img, K, min_size=(1, 1))),
             [3, 5, 9, 15, 25, 50, 75]),
            ('$WaveletFix$', (lambda img, P: WaveletFixImg2Img(1, J=1, P=P)(orig_img)),
                set(closest_waveletfix_for_blur + closest_waveletfix_for_median)),
        ]

        for method_name, fn, params in instructions:
            for param in params:
                _new_img = fn(orig_img, param)
                img_size[(method_name, param)] = list(_new_img.shape[-2:])
                _imr, _odrs = _get_odr_imr(orig_img, _new_img)
                data_odrs.setdefault((method_name, param), []).extend(_odrs)
                data_imrs.setdefault((method_name, param), _imr)
        break
    print(img_size)

    dfodr = pd.Series(
        {k: np.mean(v) for k, v in data_odrs.items()}, name='ODR')
    dfimr = pd.Series(data_imrs, name='IMR')
    dfsize = pd.Series(img_size, name='Img Shape')
    df = pd.concat([dfodr, dfimr, dfsize], axis=1).sort_index()
    df.index.set_names(['Method', 'value'], inplace=True)
    df.sort_index(inplace=True)

    # pull in accuracy information.
    df1 = plot_accuracy_blur_median(plot=False)
    df2 = pd.concat({
        'Method': df1['Method'].str.extract('(Blur|Median|\$WaveletFix\$)')[0],
        'K': df1['Experiment'].str.extract('K=(\d+)')[0],
        'P': df1['Experiment'].str.extract('P=(\d+)')[0],
        'Prediction Perf (Test Set AVG ROC AUC)': df1['Prediction Perf (Test Set AVG ROC AUC)'],
    }, axis=1)
    df2.loc[df2['Method'] == '$WaveletFix$', 'K'] = None
    df2.loc[df2['Method'] != '$WaveletFix$', 'P'] = None
    df2 = df2.melt(['Method', 'Prediction Perf (Test Set AVG ROC AUC)'],
                   ['K',  'P'], 'Param').dropna()
    df2['value'] = df2['value'].astype('int')
    df2 = df2.drop_duplicates().set_index(['Method', 'value']).sort_index()

    assert df2.shape[0] == df.shape[0], 'code error'
    df = df.join(df2.drop(columns=['Param'])).reset_index()

    # pull in ssim information
    df3a = pd.read_csv('results/plots/ssim_Blur_vs_WaveletFix.csv')
    df3b = pd.concat({
        'Method': df3a['Method'].str.extract('(Blur|Median|\$WaveletFix\$)')[0],
        'K': df3a['Experiment'].str.extract('K=(\d+)')[0],
        'P': df3a['Experiment'].str.extract('P=(\d+)')[0],
        'SSIM': df3a['SSIM'],
    }, axis=1)
    df3b.loc[df3b['Method'] == '$WaveletFix$', 'K'] = None
    df3b.loc[df3b['Method'] != '$WaveletFix$', 'P'] = None
    df3b = df3b.melt(['Method', 'SSIM'], ['K',  'P'], 'Param').dropna()
    #
    df4a = pd.read_csv('results/plots/ssim_Median_vs_WaveletFix.csv')
    df4b = pd.concat({
        'Method': df4a['Method'].str.extract('(Median|Median|\$WaveletFix\$)')[0],
        'K': df4a['Experiment'].str.extract('K=(\d+)')[0],
        'P': df4a['Experiment'].str.extract('P=(\d+)')[0],
        'SSIM': df4a['SSIM'],
    }, axis=1)
    df4b.loc[df4b['Method'] == '$WaveletFix$', 'K'] = None
    df4b.loc[df4b['Method'] != '$WaveletFix$', 'P'] = None
    df4b = df4b.melt(['Method', 'SSIM'], ['K',  'P'], 'Param').dropna()
    #
    df5 = pd.concat([df3b, df4b])
    df5['value'] = df5['value'].astype('int')
    df5 = df5.drop_duplicates().set_index(['Method', 'value']).sort_index()

    assert df5.shape[0] == df.shape[0], 'code error'
    df = df.set_index(['Method', 'value']).join(df5).reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.lineplot(data=df, x='SSIM', y='IMR', hue='Method', ax=axs[0],
                 legend=False)
    sns.lineplot(data=df, x='Prediction Perf (Test Set AVG ROC AUC)',
                 y='IMR', hue='Method', ax=axs[1], legend=False)
    sns.scatterplot(data=df, x='SSIM', y='IMR', hue='Method', ax=axs[0])
    sns.scatterplot(data=df, x='Prediction Perf (Test Set AVG ROC AUC)',
                    y='IMR', hue='Method', ax=axs[1])
    fig.tight_layout()

    savefp =
    './results/plots/imr_ssim_acc_blur_median_waveletfix.png'
    # "more private and as accurate at the same compression ratio"
    # more accurate at the same privacy level
    print('save to:', savefp)
    fig.savefig(savefp, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, y='Prediction Perf (Test Set AVG ROC AUC)',
                 x='SSIM', hue='Method', ax=ax, legend=False)
    sns.scatterplot(data=df, y='Prediction Perf (Test Set AVG ROC AUC)',
                    x='SSIM', hue='Method', ax=ax)
    fig.tight_layout()
    savefp = 'results/plots/acc_ssim_blur_median.png'
    print('save to:', savefp)
    fig.savefig(savefp, bbox_inches='tight')

    return df


def _plot_nimble(csv_files, savefp,
                 competing_method_name_in_legend, waveletfix_name_in_legend,
                 hline: Optional[Tuple[int, str]] = None, figsize=(6, 4)):
    results = []
    for lst in csv_files:
        dnn = pd.read_csv(lst[1])
        waveletfix = pd.read_csv(lst[3])
        results.append(
            [lst[0], dnn['seconds_training_epoch'].min(),
             lst[2], waveletfix['seconds_training_epoch'].min()])
    df = pd.DataFrame(results, columns=[
        'Architecture', competing_method_name_in_legend,
        'with $WaveletFix$', waveletfix_name_in_legend])  # 2nd and 4th column are SPE
    # print(df.head(1))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if hline:
        ax.hlines(hline[0], -.5, len(csv_files)-.5, label=hline[1],
                  color='gray', linestyle='dashed')
    sns.barplot(
        data=(
            df
            .melt(id_vars=['Architecture'],
                  value_vars=[competing_method_name_in_legend,
                              waveletfix_name_in_legend],
                  value_name='SPE', var_name='Experiment')
        ),
        hue='Experiment', x='Architecture', y='SPE', ax=ax)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=15, horizontalalignment='center')
    fig.tight_layout()
    fig.savefig(savefp, bbox_inches='tight')
    print('save to:', savefp)
    return df


def plot_nimble_DNNs():
    csv_files = [
        ['DenseNet121',
         'results/4.C8.diagnostic.densenet121.baseline.fromscratch/perf.csv',
         '$WaveletFix$ J=1 P=115', 'results/2.C28.J=1.P=115/perf.csv'],

        ['EfficientNet-b0', 'results/4.C8.diagnostic.efficientnet-b0.baseline/perf.csv',
         '$WaveletFix$ J=1 P=160', 'results/2.C30.J=1.P=160/perf.csv'],

        ['ResNet18', 'results/4.C8.diagnostic.resnet18.baseline.fromscratch/perf.csv',
         '$WaveletFix$ J=1 P=160', 'results/2.C25.J=1.P=160/perf.csv'],

        ['EfficientNetV2_m', 'results/4.C8.diagnostic.efficientnetv2_m.baseline/perf.csv',
         '$WaveletFix$ (J=1 P=115)', 'results/2.C33.J=1.P=115.efficientnetv2_m/perf.csv'],

        ['MDMLP_320', 'results/4.C8.diagnostic.mdmlp_320.baseline/perf.csv',
         '$WaveletFix$ (J=1 P=79)', 'results/2.C35.J=1.P=79/perf.csv'],

        ['VOLO_d1_224', 'results/4.C8.diagnostic.volo_d1_224.baseline/perf.csv',
         '$WaveletFix$ (J=1 P=115)', 'results/2.C29.J=1.P=115/perf.csv'],

        ['CoAtNet_224', 'results/4.C8.diagnostic.coatnet_1_224.baseline.adamw2/perf.csv',
         '$WaveletFix$ (J=1 P=160)', 'results/2.C36.J=1.P=160/perf.csv'],

        ['VIP_s7', 'results/4.C8.diagnostic.vip_s7.baseline/perf.csv',
         '$WaveletFix$ (J=1 P=115)', 'results/2.C33.J=1.P=115.vip_s7/perf.csv'],
        # vip_s7 not evaluated on all models.
    ]
    df = _plot_nimble(csv_files, 'results/plots/nimble_dnn_vs_waveletfix.png',
                      'DNN', '$WaveletFix$', figsize=(12, 4))
    return df


def plot_nimble_blur_median():
    blur_csv_files = [
        ['K=4', 'results/2.C31.K=4/perf.csv',
         '$WaveletFix$ (J=1 P=80)', 'results/2.C31b.J=1.P=80/perf.csv'],
        ['K=8', 'results/2.C31.K=8/perf.csv',
         '$WaveletFix$ (J=1 P=40)', 'results/2.C31b.J=1.P=40/perf.csv'],
        ['K=16', 'results/2.C31.K=16/perf.csv',
         '$WaveletFix$ (J=1 P=20)', 'results/2.C31b.J=1.P=20/perf.csv'],
        ['K=32', 'results/2.C31.K=32/perf.csv',
         '$WaveletFix$ (J=1 P=10)', 'results/2.C31b.J=1.P=10/perf.csv'],
        ['K=64', 'results/2.C31.K=64/perf.csv',
         '$WaveletFix$ (J=1 P=5)', 'results/2.C31b.J=1.P=5/perf.csv'],
    ]
    median_csv_files = [
        ['K=3', 'results/2.C32.K=3/perf.csv', '$WaveletFix$ (J=1 P=80)',
         'results/2.C31b.J=1.P=80/perf.csv'],
        ['K=5', 'results/2.C32.K=5/perf.csv', '$WaveletFix$ (J=1 P=80)',
         'results/2.C31b.J=1.P=80/perf.csv'],
        ['K=9', 'results/2.C32.K=9/perf.csv', '$WaveletFix$ (J=1 P=40)',
         'results/2.C31b.J=1.P=40/perf.csv'],
        ['K=15', 'results/2.C32.K=15/perf.csv', '$WaveletFix$ (J=1 P=40)',
         'results/2.C31b.J=1.P=40/perf.csv'],
        ['K=25', 'results/2.C32.K=25/perf.csv', '$WaveletFix$ (J=1 P=18)',
         'results/2.C31b.J=1.P=18/perf.csv'],
        ['K=50', 'results/2.C32.K=50/perf.csv', '$WaveletFix$ (J=1 P=6)',
         'results/2.C31b.J=1.P=6/perf.csv'],
        ['K=75', 'results/2.C32.K=75/perf.csv',
            '$WaveletFix$ (J=1 P=4)', 'results/2.C31b.J=1.P=4/perf.csv'],
    ]
    df1 = _plot_nimble(
        median_csv_files, 'results/plots/nimble_median_vs_waveletfix.png',
        'Median, K={K}', 'closest $WaveletFix$, J=1, P={P}',
        hline=(107.644, 'DNN (EfficientNet-b0, no privatized compression)'))  # number from 4.C8 efficientnet SPE.min()
    df2 = _plot_nimble(
        blur_csv_files, 'results/plots/nimble_blur_vs_waveletfix.png',
        'Blur, K={K}', 'closest $WaveletFix$, J=1, P={P}',
        hline=(107.644, 'DNN (EfficientNet-b0, no privatized compression)'))  # number from 4.C8 efficientnet SPE.min()
    return (df1, df2)


if __name__ == '__main__':
    plot_accuracy_DNNs()
    plot_accuracy_blur_median()
    plot_ssim_blur_median()
    plot_compression_blur_median()
    plot_nimble_DNNs()
    plot_nimble_blur_median()

    # this was used to find the J and P comparable to blur and median.
    # and plot the side-by-side images of the blur and closest waveletfix
    find_waveletfix_model_most_comparable_to(
        blur, kernel_sizes=[4, 8, 16, 32, 64])
    find_waveletfix_model_most_comparable_to(
        median, kernel_sizes=[3, 5, 9, 15, 25, 50, 75])
