import torch as T
from simple_parsing import ArgumentParser, choice
from os import makedirs
from dataclasses import dataclass
from deepfixcx.train import TrainOptions, train_config
from deepfixcx import init_from_distribution as DI
from deepfixcx import weight_saliency as W


def reinit_model(model:T.nn.Module, bn=False):
    for name, param in model.named_parameters():
        layer = model.get_submodule(name.rsplit('.', 1)[0])
        if isinstance(layer, (T.nn.modules.conv._ConvNd, T.nn.Linear)):
            DI.reinitialize_parameters_(name, layer, param)
        elif isinstance(layer, T.nn.modules.batchnorm._NormBase):
            pass
        else:
            raise NotImplementedError(f'how to reinitialize parameter {name}?')


def plot_histograms(histograms, mode, model, save_dir=None):
    from matplotlib import pyplot as plt
    fig1, axs1 = plt.subplots(6,4, figsize=(12, 8))
    fig2, axs2 = plt.subplots(6,4, figsize=(12, 8))
    fig3, axs3 = plt.subplots(6,4, figsize=(12, 8))
    for key, ax1, ax2, ax3 in zip(histograms.keys(), axs1.reshape(-1), axs2.reshape(-1), axs3.reshape(-1)):
        counts, bin_edges = histograms[key]
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        for i in range(min(5, len(counts))):
            #  ax1.scatter(bin_edges.cpu().numpy(), counts[i].cpu().numpy(), marker='.')
            #  ax2.scatter(bin_edges.cpu().numpy(), counts[-1*i].cpu().numpy(), marker='.')
            ax1.plot(bin_centers.cpu().numpy(), counts[i].cpu().numpy())
            ax2.plot(bin_centers.cpu().numpy(), counts[-1*i].cpu().numpy())
            ax3.plot(bin_centers.cpu().numpy(), counts[len(counts)//2-2+i].cpu().numpy())
        [ax.set_title(key) for ax in [ax1, ax2]]
    fig1.suptitle(f'Distribution of Most Salient Weights\nAcross Layers of {args.TrainOptions.model}')
    fig2.suptitle(f'Distribution of Least Salient Weights\nAcross Layers of {args.TrainOptions.model}')
    fig3.suptitle(f'Distribution of Median Salient Weights\nAcross Layers of {args.TrainOptions.model}')
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    if save_dir:
        fps = [
            f'{save_dir}/dist_first_{mode}_{model}.png',
            f'{save_dir}/dist_last_{mode}_{model}.png',
            f'{save_dir}/dist_middle_{mode}_{model}.png'
        ]
        makedirs(save_dir, exist_ok=True)
        for fp, fig in zip(fps, [fig1, fig2, fig3]):
            fig.savefig(fp, bbox_inches='tight')
            print(fp)


if __name__ == "__main__":
    @dataclass
    class Options:
        mode:str = choice('nth_most_salient', 'nth_weight', default='nth_most_salient')
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
        model:str = 'resnet18:untrained:3:3'

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
    ranges = None  # {'param_name': Optional[int]}  # bounds (e.g. range) of the histogram.  assume pre-training and fine-tuning decreases the bound.
    for i in range(args.Options.iters):
        print('iter', i)
        if ranges is None:
            ranges = DI.get_kaiming_uniform_bounds(mdl)
        reinit_model(mdl)
        # observe saliency of weights
        sr:W.SaliencyResult = W.get_saliency(
            cost_fn=W.costfn_multiclass, model=mdl, loader=loader,
            device=device, num_minibatches=100,  # TODO: 1 or 100 minibatches?
            mode=args.Options.saliency_mode
        )
        # update a per-weight distribution
        for layer_idx,psr in enumerate(sr):
            if ranges[psr.param_name] is None:
                #  print(f'skip layer: {psr.param_name}')
                continue
            try:
                counts, bin_edges = histograms[psr.param_name]
            except KeyError:  # initialize histogram first time
                bins = 50
                # TODO: for paper, plot the max of uniform random Conv2d vs pre-trained.
                counts = T.zeros((psr.weight.numel(), bins), dtype=T.float, device='cpu')
                left, right = ranges[psr.param_name]
                # --> augment boundaries to account for floating point error
                eps = 1e-4
                left, right = left-eps, right+eps
                # --> maintain extra bin to right and left to check there are no values outside the bound
                # the left-most and right-most bins should always count to zero
                bin_edges = T.linspace(left, right, bins+1, device=device)
                histograms[psr.param_name] = counts, bin_edges
            chosen_bin = T.bucketize(psr.weight.view(-1), boundaries=bin_edges)
            a,b,c,d = (psr.weight.min().item(), bin_edges[0].item(), psr.weight.max().item(), bin_edges[-1].item())
            assert a>b and c<d, 'sanity check'
            assert chosen_bin.min() > 0, 'sanity check'
            assert chosen_bin.max() < bins+1, 'sanity check'
            # --> now that we passed checks, shift bins so it is zero indexed
            # --> there are N+1 bin "edges" and N bins (i.e. counts)
            # --> bin edges:  [ -1,  0,  1]
            #     counts:        [c1, c2]
            chosen_bin = chosen_bin - 1
            if args.Options.mode == 'nth_most_salient':
                chosen_bin = chosen_bin[T.argsort(psr.saliency.view(-1), descending=True)]
            chosen_bin = chosen_bin.to('cpu', non_blocking=True)
            saliency = psr.saliency.view(-1).to('cpu') #, non_blocking=True)

            rowidxs = T.arange(counts.shape[0])
            if args.Options.mode == 'nth_most_salient':
                a = counts.sum()
                counts.scatter_add_(1, chosen_bin.reshape(-1,1), T.ones_like(counts))
                b = counts.sum()
                assert T.allclose(b, a + len(rowidxs)), 'sanity check: updating array'
            elif args.Options.mode == 'nth_weight':
                a = counts.sum()
                counts.scatter_add_(1, chosen_bin.reshape(-1,1), saliency.view(-1,1).expand_as(counts))
                b = counts.sum()
                assert T.allclose(b, a + saliency.sum()), 'sanity check: updating array'

    # sanity check results
    if args.Options.mode == 'nth_most_salient':
        cs = counts.cumsum(1)[:, -1]
        assert T.allclose(cs[0], cs), 'sanity check: error updating the array.  all histograms should sum to the same value'

    # save histograms to disk
    makedirs(f'{args.Options.base_dir}/histograms', exist_ok=True)
    fp_hist = f'{args.Options.base_dir}/histograms/hist_{args.Options.mode}.ws{args.Options.saliency_mode}_{args.TrainOptions.model}.pth'
    T.save(histograms, fp_hist)
    print(fp_hist)
    plot_histograms(
        histograms,
        args.Options.mode+ f".ws{args.Options.saliency_mode.replace('*', '')}", args.TrainOptions.model,
        save_dir=f'{args.Options.base_dir}/histograms/plots')



"""

TODO:
    - Summarize findings of I3 tests
      - outperform fromscratch baseline, but not imagenet baseline.

      - not screwing with batchnorm.  should we inherit imagenet batchnorm weights to get better perf?
      - num_minibatches=1.  maybe that was too small?
      - performance of fixed models sucks (to be verified).
      - which saliency is better?  grad, weight, or weight*grad ?

    - Possible "I4" tests:
        - use the hist to only initialize model with most salient weights.
          --> if works on any models (e.g. the wsweight model), then the
              distribution might tell us we should we avoid initializing near
              zero, and instead sample values close to the bounds.
        - only initialize with most / least salient weights (but no middle weights)
        - only initialize with middle weights

    - how to capture the distribution of imagenet weights?  (guidedsteer)


    - check that the I3 tests didn't outperform the


        """


"""
todo (old)

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

        # Q: Are the weights different in distribution?
        #


shreshta mtg notes:


redundancy vs saliency:

  - redundant weights can be removed.
  - less salient


- try maintaining distribution
  - of most salient nodes (or quantiles of all nodes)
  - of all neurons
  """
