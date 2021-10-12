import torch as T
from typing import Optional
from simple_parsing import ArgumentParser
from deepfix.train import TrainOptions, train_config
from deepfix import weight_saliency as W


def get_cfg():
    p = ArgumentParser()
    p.add_arguments(TrainOptions, dest='TrainOptions')
    args = p.parse_args([
        '--device', 'cuda', '--model', 'resnet18:untrained:3:3']).TrainOptions
    cfg = train_config(args)
    return cfg, args


def get_ranges(model:T.nn.Module) -> dict[str,Optional[tuple[float,float]]]:
    """
    For the purpose of storing histograms for each of the model's named
    parameters, compute the range of each histogram.  For all weights
    associated to a parameter, the range is the same.
    """
    rv = {}
    for param_name, param in model.named_parameters():
        layer = model.get_submodule(param_name.rsplit('.', 1)[0])
        if isinstance(layer, (T.nn.modules.conv._ConvNd, T.nn.Linear)):
            # compute bound of kaiming uniform U[-b,b]
            bound = W.get_kaiming_uniform_bound(layer.weight.data)
            bound = (-1.*bound, 1.*bound)
        elif isinstance(layer, T.nn.modules.batchnorm._BatchNorm):
            bound = None
        else:
            raise NotImplementedError(f'How to compute the bound on this parameter:  {param_name}')
        rv[param_name] = bound
    return rv


def reinit_model(model:T.nn.Module, bn=False):
    for name, param in model.named_parameters():
        layer = model.get_submodule(name.rsplit('.', 1)[0])
        if isinstance(layer, (T.nn.modules.conv._ConvNd, T.nn.Linear)):
            W.reinitialize_parameters_(name, layer, param)
        elif isinstance(layer, T.nn.modules.batchnorm._NormBase):
            pass
        else:
            raise NotImplementedError(f'how to reinitialize parameter {name}?')


def plot_histograms(histograms, mode, model):
    from matplotlib import pyplot as plt
    fig1, axs1 = plt.subplots(6,4, figsize=(12, 8))
    fig2, axs2 = plt.subplots(6,4, figsize=(12, 8))
    for key, ax1, ax2 in zip(histograms.keys(), axs1.reshape(-1), axs2.reshape(-1)):
        counts, bin_edges = histograms[key]
        for i in range(min(5, len(counts))):
            ax1.scatter(bin_edges.cpu().numpy(), counts[i].cpu().numpy(), marker='.')
            ax2.scatter(bin_edges.cpu().numpy(), counts[-1*i].cpu().numpy(), marker='.')
        [ax.set_title(key) for ax in [ax1, ax2]]
    fig1.suptitle(f'Distribution of Most Salient Weights\nAcross Layers of {args.model}')
    fig2.suptitle(f'Distribution of Least Salient Weights\nAcross Layers of {args.model}')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f'dist_first_{mode}_{model}.png', bbox_inches='tight')
    fig2.savefig(f'dist_last_{mode}_{model}.png', bbox_inches='tight')


if __name__ == "__main__":
    cfg, args = get_cfg()
    loader = cfg.train_loader
    device = cfg.device
    mdl = cfg.model
    del cfg

    mode = 'nth_most_salient'
    #  mode = 'nth_weight'

    histograms = {}  # dict[int:np.ndarray]  of {'param_name': 'counts of shape (numel,bins)'}
    ranges = None  # {'param_name': Optional[int]}  # bounds (e.g. range) of the histogram.  assume pre-training and fine-tuning decreases the bound.
    num_iter = 100  # 500  # num models for the probability distribution
    for i in range(num_iter):
        print('iter', i)
        if ranges is None:
            ranges = get_ranges(mdl)
        reinit_model(mdl)
        # initialize with larger bounds:
        #  for name, param in mdl.named_parameters():
        #      if ranges[name] is not None:
        #          param.data *= 400
        #          a,b = ranges[name]
        #          ranges[name] = 400.1*a, 400.1*b
        # observe saliency of weights
        sr:W.SaliencyResult = W.get_saliency(
            cost_fn=W.costfn_multiclass, model=mdl, loader=loader,
            device=device, num_minibatches=1,  # TODO: more minibatches?
        )
        # update a per-weight distribution
        for layer_idx,psr in enumerate(sr):
            if ranges[psr.param_name] is None:
                #  print(f'skip layer: {psr.param_name}')
                continue

            saliency = psr.saliency.view(-1).to('cpu') #, non_blocking=True)
            try:
                counts, bin_edges = histograms[psr.param_name]
            except KeyError:  # initialize histogram first time
                bins = 51
                # TODO: for paper, plot the max of uniform random Conv2d vs pre-trained.
                counts = T.zeros((psr.weight.numel(), bins), dtype=T.float, device='cpu')
                bin_edges = T.linspace(*ranges[psr.param_name], bins, device=device)
                histograms[psr.param_name] = counts, bin_edges
            chosen_bin = T.bucketize(psr.weight.view(-1), boundaries=bin_edges[:-1])
            a,b,c,d = (psr.weight.min().item(), bin_edges[0].item(), psr.weight.max().item(), bin_edges[-1].item())
            assert a>=b and c<=d
            if mode == 'nth_most_salient':
                chosen_bin = chosen_bin[T.argsort(psr.saliency.view(-1), descending=True)]
            chosen_bin = chosen_bin.to('cpu', non_blocking=True)

            rowidxs = T.arange(counts.shape[0])
            if mode == 'nth_most_salient':
                counts[rowidxs, chosen_bin] += 1
            elif mode == 'nth_weight':
                counts[rowidxs, chosen_bin] += saliency.view(-1)

    # save histograms to disk
    T.save(histograms, f'hist_{mode}_{args.model}.pth')
    plot_histograms(histograms, mode, args.model)



""" todo next

re-init model by sampling from hist.
evaluate the model


    # hist to probability
    # sample values from histograms.  do we need to smooth them first?
    for key, w in mdl.named_parameters():
        sampled_from_hist = histograms[key] + ... # need more code
        # TODO gaussian kde of hist.  sample it.
        w.data[:] = sampled_from_hist

        # TODO: could trivially parallelize the for loops
        # TODO: computing the distribution: brute force histogram, sketching approximate distribution,
        # TODO: try using the distribution with hyperband to select new initializations
        # TODO: distribution of either weights by saliency (e.g. quantile) or weights by size.

        TODO: eval saliency with only gradient vs gradient*weight.


        # Q: Are the weights different in distribution?
        #
        """


"""

shreshta mtg notes:


redundancy vs saliency:

  - redundant weights can be removed.
  - less salient


- try maintaining distribution
  - of most salient nodes (or quantiles of all nodes)
  - of all neurons
  """
