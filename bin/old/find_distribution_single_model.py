import torch as T
from typing import Optional
from simple_parsing import ArgumentParser, choice
from os import makedirs
from dataclasses import dataclass
from matplotlib import pyplot as plt
from waveletfix.train import TrainOptions, train_config
from waveletfix import weight_saliency as W


def plot(histograms:dict[str,T.Tensor]):
    fig, axs = plt.subplots(8,8, figsize=(16, 12), sharex=True)
    axs[0,0].set_xlim(-1,1)
    for param_name, ax in zip(histograms, axs.ravel()):
        counts, bin_edges = histograms[param_name]
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        ax.plot(bin_centers.cpu().numpy(), counts.cpu().numpy())
        ax.set_title(param_name)
    fig.tight_layout()
    return fig


def save(histograms:dict[str,T.Tensor], fp):
    pass


if __name__ == "__main__":
    @dataclass
    class Options:
        saliency_mode:str = choice('weight*grad', 'grad', 'weight', default='weight*grad')
        iters:int = 500
        base_dir:str = './results/debugging'

    @dataclass
    class SecondaryConfig(TrainOptions):
        """
        Used for convenience in order to access things like dataloader.
        No training actually takes place.
        """
        device:str = 'cuda'
        model:str = 'resnet18:imagenet:3:3'

    par = ArgumentParser()
    par.add_arguments(Options, dest='Options')
    par.add_arguments(SecondaryConfig, dest='TrainOptions')
    args = par.parse_args()

    cfg = train_config(args.TrainOptions)
    loader = cfg.train_loader
    device = cfg.device
    mdl = cfg.model.eval()
    del cfg

    histograms = {}  # dict[int:np.ndarray]  of {'param_name': 'counts of shape (numel,bins)'}

    # observe saliency of weights
    sr:W.SaliencyResult = W.get_saliency(
        cost_fn=W.costfn_multiclass, model=mdl, loader=loader,
        device=device, num_minibatches=1,  # TODO: 1 or 100 minibatches?
        mode=args.Options.saliency_mode
    )

    histograms = {}
    for layer_idx,psr in enumerate(sr):
        if args.Options.saliency_mode == 'weight':
            saliency = psr.weight.view(-1)  # hack: use raw value, not magnitude
        else:
            saliency = psr.saliency.view(-1)
        saliency = mdl.get_parameter(psr.param_name).view(-1).detach()
        left, right = saliency.quantile(T.tensor([0,1.], device=device))
        bin_edges = T.linspace(left-1e-5, right+1e-5, 100, device=device)
        bins = T.searchsorted(bin_edges, saliency)  # b bin_edges --> b+1 bins
        counts = T.bincount(bins, minlength=101)
        assert counts[0] == 0 and counts[-1] == 0, 'bug: bin edges out of range'
        counts = counts[1:-1]
        histograms[psr.param_name] = (counts, bin_edges)

    O = args.Options
    TO = args.TrainOptions
    makedirs(f'{O.base_dir}/plots', exist_ok=True)
    fp_plot = f'{O.base_dir}/plots/{TO.model}_{O.saliency_mode.replace("*","")}.png'
    plot(histograms).savefig(fp_plot, bbox_inches='tight')
    fp_hist = f'{O.base_dir}/{TO.model}_{O.saliency_mode.replace("*","")}.pth'
    T.save(histograms, fp_hist)
    print(fp_plot)
    print(fp_hist)
