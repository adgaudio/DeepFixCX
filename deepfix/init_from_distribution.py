from simple_parsing import ArgumentParser
from typing import Optional, Callable
import numpy as np
import math
import torch as T
from deepfix import weight_saliency as W


def init_from_beta(model: T.nn.Module, alpha, beta):
    """
    Doesn't work - don't use it.

    Use the same bounds that the Kaiming Uniform distribution uses, but sample
    only values close to the boundaries.  Do this using a Beta distribution.
    When both of Beta distribution parameters are less than 1, the distribution
    is U-shaped and support is [0,1].

    The idea is to address the switch effect by switching all nodes "on" at the
    start.

    The idea of this initialization is to modify Kaiming Uniform initialization:

        x ~ U[a,b]  (kaiming uniform initialization)
        x ~ Beta[.3, .3] * (b-a) - a  (our initialization)

        Shares same bounds [a,b].  Doesn't share same variance.

    """
    pdist = T.distributions.Beta(alpha, beta)
    bounds = get_kaiming_uniform_bounds(model)
    for param_name, param in model.named_parameters():
        layer_inst = model.get_submodule(param_name.rsplit('.', 1)[0])
        if isinstance(layer_inst, (T.nn.modules.conv._ConvNd, T.nn.Linear)):
            a,b = bounds[param_name]
            param.data[:] = pdist.sample(param.shape) * (b-a) + a
        #  else:
            #  print('skip', layer_inst)


def init_from_hist_(model:T.nn.Module, hist:dict[str, (T.Tensor, T.Tensor)]):
    """
    Initialize model weights with values from the histogram.
    Modify model inplace.

    Args:
        model: a pytorch model with named parameters
        hist: a dict of histograms of form {"named_parameter": (counts, bin_edges)}
            where "named_parameter" identifies a parameter
            (i.e. from `model.named_parameters()`),
            and where `counts` and `bin_edges` define one histogram for each scalar
            value in the parameter.
            - `counts` is of shape (N, M) where N is the number of scalar
                values in the parameter, and M is the bin size.
            - `bin_edges` is of shape (N+1) denoting the beginning and end of
              each histogram bin.
    """
    for param_name in hist:
        counts, bin_edges = hist[param_name]
        if counts.ndim == 1:
            counts = counts.reshape(1,-1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        param = model.get_parameter(param_name)
        # convert histogram to cumulative probability distribution
        cs = counts.cumsum(1).float()
        cs /= cs[:,[-1]]
        assert T.allclose(cs[:,-1], T.tensor(1.))
        # for each weight parameter, sample a value from the distribution
        sample = T.rand((param.numel(), 1), device=cs.device)
        csbins = T.searchsorted(cs.expand((sample.shape[0], cs.shape[1])), sample)
        values = bin_centers[csbins]
        # add noise, staying within boundary of a bin
        bin_width = bin_edges[1] - bin_edges[0]
        noise = T.rand(counts.shape[0], 1, device=bin_width.device) * bin_width/2
        values += noise
        param.data[:] = values.reshape(param.shape)


def reset_optimizer(opt_spec:str, model:T.nn.Module) -> T.optim.Optimizer:
    spec = opt_spec.split(':')
    kls = getattr(T.optim, spec[0])
    params = [(x,float(y)) for x,y in [kv.split('=') for kv in spec[1:]]]
    optimizer = kls(model.parameters(), **dict(params))
    return optimizer


def get_kaiming_uniform_bounds(model:T.nn.Module) -> dict[str,Optional[tuple[float,float]]]:
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
            bound = get_kaiming_uniform_bound(layer.weight.data)
            bound = (-1.*bound, 1.*bound)
        elif isinstance(layer, T.nn.modules.batchnorm._BatchNorm):
            bound = None
        else:
            raise NotImplementedError(f'How to compute the bound on this parameter:  {param_name}')
        rv[param_name] = bound
    return rv


def get_kaiming_uniform_bound(tensor, gain=math.sqrt(1./3), mode:str='fan_in'):
    """
    By default, compute `1/sqrt(fan)` where `fan = tensor.shape[1] * prod(tensor.shape[2:])`
    Args:
        mode:
            'fan_in' preserves magnitude of variance of weights in forward pass,
            'fan_out' preserves magnitude of variance of weights in backward pass.
            Pytorch uses fan_in by default.
        gain: Set to
            sqrt(1/3) if assume "leaky_relu" is the non-linearity (default in pytorch)
            sqrt(2) if "relu" (pytorch recommends "leaky_relu")
            1 if assuming linear fn or sigmoid.
    """
    assert mode in {'fan_in', 'fan_out'}, f'user error: invalid mode {mode}'
    fan = math.prod(tensor.shape[2:]) * ( tensor.shape[0] if mode == 'fan_out' else tensor.shape[1])
    bound = math.sqrt(3.) * gain / math.sqrt(fan)
    return bound


def reinitialize_parameters_(param_name:str, layer_inst:T.nn.Module, randvals:T.Tensor, return_bound=False):
    """Re-initialize `randvals` in place and return it.  Try to use the default
    initialization available in pytorch."""
    if isinstance(layer_inst, (T.nn.modules.conv._ConvNd, T.nn.Linear)):
        if param_name.endswith('.weight') or param_name.endswith('.bias'):
            # same result as what pytorch does by default, but clearer:
            bound = get_kaiming_uniform_bound(layer_inst.weight.data)
            T.nn.init.uniform_(randvals, -1.*bound, bound)
        else:
            raise NotImplementedError()
    elif isinstance(layer_inst, T.nn.modules.batchnorm._NormBase):
        bound = None
        if param_name.endswith('.weight'):
            T.nn.init.ones_(randvals)
        elif param_name.endswith('.bias'):
            T.nn.init.zeros_(randvals)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(f"How to re-initialize param {param_name}")
    if return_bound:
        return randvals, bound
    else:
        return randvals


def get_flat_view(s: W.SaliencyResult):
    s2s = T.cat([x.view(-1) for x in s.saliency])
    s2w = T.cat([x.view(-1) for x in s.weights])
    s2n = np.array([name for name, w in zip(s.param_names, s.weights)
                    for _ in range(w.numel())])
    return W.SaliencyResult(s2s, s2n, s2w)


Y, YHat, Scalar = T.Tensor, T.Tensor, T.Tensor  # for type checking


def reinitialize_least_salient(
        cost_fn: Callable[('Y', 'YHat'), 'Scalar'],
        model:T.nn.Module, loader:T.utils.data.DataLoader,
        device:str, M:int, frac:float, opt:Optional[T.optim.Optimizer],
        reinitialize_fn:Callable[[str,T.nn.Module],T.Tensor]=reinitialize_parameters_
):
    """Re-initialize the `frac` amount of least salient weights in the model with random values.

    Args:
        - `M` minibatch size for saliency
        - `frac` fraction of weights with smallest saliency scores in [0,1]
        - `reinitialize_fn`: A function that inplace-updates the 3rd parameter
        (a tensor) with re-initialized values.
            Fn Inputs: param_name:str, layer_inst:T.nn.Module, param:T.Tensor
            The param tensor will be inplace modified by this fn.
    """
    s = W.get_saliency(
        cost_fn=cost_fn,
        model=model, loader=loader, device=device,
        num_minibatches=M)
    sflat = get_flat_view(s)
    # --> get bottom k least salient weights
    k = int(frac*len(sflat.saliency))
    s_lo = T.topk(sflat.saliency, k, largest=False, sorted=False).indices.sort().values
    # --> group all weights by layer.
    _idx_bins = np.roll(np.cumsum([x.numel() for x in s.weights]), 1)
    _idx_bins[0] = 0
    paramname_to_idx = {paramname: _idx_bins[v] for v, paramname in enumerate(s.param_names)}
    group_by_paramname = T.bucketize(
        s_lo, T.tensor(_idx_bins, device=s_lo.device), right=True)
    # --> Update the (globally) least salient weights across all paramnames
    for paramname in group_by_paramname.unique():
        mask = group_by_paramname == paramname
        flat_idxs = s_lo[mask]
        paramname = [sflat.param_names[i] for i in flat_idxs]
        assert all(x == paramname[0] for x in paramname)
        paramname = paramname[0]
        param = model.get_parameter(paramname)
        l_idx = paramname_to_idx[paramname]
        x = np.unravel_index([flat_idxs.cpu().numpy()-l_idx], param.shape)
        # re-initialize weights
        layer_inst = model.get_submodule(paramname.rsplit('.', 1)[0])
        randvals = reinitialize_fn(paramname, layer_inst, T.clone(param.data))
        param.data[x] = randvals[x]
        # optionally, modify the optimizer buffer
        # TODO: Only tested on SGD with momentum.
        # TODO: not sure this improves perf, but seems like it might.
        if opt is not None:
            z = opt.state[param]
            if 'momentum_buffer' in z:
                assert z['momentum_buffer'].shape == param.shape, 'sanity check'
                z['momentum_buffer'][x] = 0
                assert len(z) == 1, f"Not Implemented: optimizer has other parameters might need to re-initialize: {z.keys()}"
            else:
                assert len(z) == 0, f'Not Implemented:  optimizer has other parameters might need to re-initialize: {z.keys()}'


def test_reinitialize_parameters_():
    # test kaiming uniform bounds U[-bound, bound] is correct for all layer types
    layers = [
        T.nn.Conv2d(100,200,3,3),
        T.nn.Conv2d(300,10,1,1),
        T.nn.Linear(123,45),
        T.nn.Linear(123,456),
        T.nn.Linear(123,123),]
    for z in layers:
        # empirical bounds
        pytorch_bounds = [0,0]
        for _ in range(400):
            z.reset_parameters()
            b = pytorch_bounds
            pytorch_bounds = [max(b[0], z.weight.data.abs().max().item()),
                              max(b[1], z.bias.data.abs().max().item())]
        # exact bounds
        our_bounds = [
            reinitialize_parameters_(
                '.weight', z, z.weight.data.clone(), return_bound=True)[1],
            reinitialize_parameters_(
                '.bias', z, z.bias.data.clone(), return_bound=True)[1]
        ]
        our_empirical_bounds = [
            reinitialize_parameters_(
                '.weight', z, z.weight.data.clone(), return_bound=True)[0].abs().max(),
            reinitialize_parameters_(
                '.bias', z, z.bias.data.clone(), return_bound=True)[0].abs().max()
        ]
        assert (T.tensor(our_bounds) >= T.tensor(pytorch_bounds)).all(), \
            'Our reinitialization bounds do not match pytorch default. They are too small.'
        assert ((T.tensor(our_bounds) - T.tensor(pytorch_bounds)).abs().max() < .0001), \
            'Our reinitialization bounds are too far away from pytorch default.'
        assert (T.Tensor(our_bounds) >= T.tensor(our_empirical_bounds)).all(), 'code bug'
        assert T.allclose(T.Tensor(our_bounds), T.tensor(pytorch_bounds), 5e-2, 5e-2), 'mismatch pytorch'
    print('reinitialization test: reinitialization does same as pytorch passes')


if __name__ == "__main__":
    test_reinitialize_parameters_()
    import sys
    from deepfix.train import TrainOptions, train_config


    def get_cfg():
        p = ArgumentParser()
        p.add_arguments(TrainOptions, dest='TrainOptions')
        args = p.parse_args([
            '--device', 'cuda', '--model', 'resnet18:untrained:3:3']).TrainOptions
        cfg = train_config(args)
        return cfg, args
    # load model
    cfg, args = get_cfg()

    fp = sys.argv[1]  # results/1.I3/histograms/hist_nth_most_salient_resnet18:untrained:3:3.pth
    fp_out = sys.argv[2]  # results/1.I3/hist_nth_most_salient_resnet18:untrained:3:3.csv
    hist = T.load(fp)

    perfs = []
    #  perfs.append(cfg.evaluate_perf(cfg))
    #  print('initialize model')
    #  init_from_hist_(cfg.model, hist)
    #  cfg.optimizer = reset_optimizer(args.opt, cfg.model)
    print('evaluate model')
    perfs.append(cfg.evaluate_perf(cfg, loader=cfg.train_loader))
    for _ in range(5):
        cfg.train_one_epoch(cfg)
        perfs.append(cfg.evaluate_perf(cfg))
        print(perfs[-1])

    import pandas as pd
    df = pd.DataFrame(perfs)
    print(df.to_string())
    df.to_csv(fp_out, index=False)





    #  T.load('./hist_nth_weight_resnet18:untrained:3:3.pth
    # eval, train+eval
