import math
import numpy as np
import torch as T
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from typing import Callable

import torchvision.models as tvm
import torchvision.transforms as tvt
from efficientnet_pytorch import EfficientNet

from simplepytorch.datasets import CheXpert_Small
from waveletfix.weight_saliency import get_saliency
from waveletfix.analyze_layer import analyze_model_at_modules


def plot_as_subplots(lst: list['data'], ax_titles:list[str]=None,
                     fig_axs:(plt.Figure, list[plt.Axes])=None,
                     plot_fn:Callable[[plt.Axes, 'data'], None]=lambda ax, data: ax.plot(data)):
    """Generic way to make subplots"""
    if fig_axs is None:
        ncol, nrow = int(math.ceil(math.sqrt(len(lst)))), int(math.sqrt(len(lst)))
        fig, axs = plt.subplots(nrow, ncol, figsize=(nrow*3, ncol*3))
    else:
        fig, axs = fig_axs
    if not ax_titles:
        ax_titles = range(len(lst))
    for weight, title, ax in zip(lst, ax_titles, axs.ravel()):
        plot_fn(ax, weight)
        ax.set_title(title)
    # fig.legend()
    fig.tight_layout()
    return fig


def get_model_and_saliency_data(mode='all_layers'):
    '''
    mode: {'pointwise_layers', 'all_layers'}
    '''
    # obtain a model.
    # model:  your best/favorite chexpert efficientnet here.
    #  model, model_name = tvm.resnet18(pretrained=True), 'resnet18:imagenet:1:1000'
    model, model_name = (
        EfficientNet.from_pretrained('efficientnet-b0', advprop=True),
        'efficientnet-b0:imagenetadv:1:1000')
    model.eval()
    # hack: make the model have 1 input channel.  try to hackishly be general
    # enough for all CNN models.  (only useful for saliency)
    for x in model.modules():
        if isinstance(x, T.nn.Conv2d):
            assert x.in_channels == 3, "hmm maybe this hack doesn't apply here"
            x.in_channels = 1
            x.weight = T.nn.Parameter(x.weight.data[:, [2]])
            break

    # plot weight vs saliency
    loader = T.utils.data.DataLoader(CheXpert_Small(
        getitem_transform=lambda dct: (
            tvt.Compose([tvt.Resize(224), tvt.CenterCrop(224)])(dct['image']),
            CheXpert_Small.format_labels(dct, labels=CheXpert_Small.LABELS_DIAGNOSTIC_LEADERBOARD),
        )), batch_size=4, shuffle=True)
    device = 'cuda'
    if mode == 'pointwise_weights':
        param_names = [
            x for x,_ in model.named_parameters()
            if x.endswith('.weight')
            and (conv2d:=model.get_submodule(x.rsplit('.', 1)[0]))
            and isinstance(conv2d, T.nn.Conv2d) and conv2d.kernel_size == (1,1)
        ]
    elif mode == 'all_layers':
        param_names = set(dict(model.named_parameters()))
    else:
        raise NotImplementedError()
    saliency = get_saliency(
        lambda y, yhat: yhat.sigmoid().sum(), model, loader, device,
        mode='weight*grad', num_minibatches=10, param_names=param_names)
    return model, model_name, saliency


if __name__ == "__main__":
    model, model_name, saliency_result = get_model_and_saliency_data('pointwise_weights')

    # consider only first 8 pointwise convs for simpler plotting.
    # (early layers are more important anyways)
    params = [x.saliency.cpu().numpy() for x in saliency_result][:8]
    param_names = [x.param_name for x in saliency_result][:8]

    import sys ; sys.exit()

    # plot whole weight matrix
    # looks like some columns are more "used" than others.
    fig = plot_as_subplots(
        params, param_names,
        plot_fn=lambda ax, w: ax.imshow(w.squeeze(), cmap='RdBu', norm=plt.cm.colors.CenteredNorm()))

    # plot histogram per column (i.e. per input channel)
    # hmm - that doesn't show it
    fig = plot_as_subplots(
        params, param_names,
        plot_fn=lambda ax, weights: sns.kdeplot(data=np.abs(weights.squeeze()), ax=ax, alpha=.3))
    fig.suptitle('Weights of each input channel, as KDE plot')

    # plot: relative importance of input channel to each output (not correcting for input channel magnitude)
    #       how:  in each row, rank columns.  Then for each column, compute histogram.
    fig = plot_as_subplots(
        params, param_names,
        #  plot_fn=lambda ax, weights: sns.ecdfplot(data=rankdata(weights.squeeze(), axis=1), ax=ax, alpha=.5))
        plot_fn=lambda ax, weights: sns.kdeplot(data=rankdata(weights.squeeze(), axis=1), ax=ax, alpha=.5))

    # 
    #  def hist_per_col(ax, weights):
    #      weights = weights.squeeze()
    #      for col in range(weights.shape[1]):
    #          counts, bin_edges = np.histogram(
    #              weights[:,col], bins=100, range=(weights.min(), weights.max()))
    #          bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    #          ax.plot(bin_centers, counts, label=col)
    #          ax.set_xlabel('weight value (100 bins)')
    #          ax.set_xlabel('count')
    #          #  ax.legend()

    # plot how 'used' each input channel is (not correcting for input channel magnitude)
    fns = [
        (lambda x: x.squeeze().sum(0), 'w.sum(0)'),
        (lambda x: np.abs(x.squeeze()).max(0), 'w.abs().max(0)'),
        (lambda x: np.abs(x.squeeze()).sum(0), 'w.abs().sum(0)'),
    ]
    for agg_fn, fn_name in fns:
        fig = plot_as_subplots(
            params, param_names,
            #  plot_fn=lambda ax, w: sns.ecdfplot(x=agg_fn(w), ax=ax))
            plot_fn=lambda ax, w: sns.histplot(x=agg_fn(w), ax=ax))
            #  plot_fn=lambda ax, w: ax.plot(agg_fn(w)))
            #  plot_fn=lambda ax, w: sns.scatterplot(x=np.abs(w.squeeze()).max(0), y=np.abs(w.squeeze()).sum(0), ax=ax))
        fig.suptitle(f'Input Channel Weights, {fn_name} for {model_name}')
        fig.tight_layout()

    #  plot_as_subplots(*,
                     #  plot_fn=lambda ax, sr: sr.weight.squeeze())


