import torch as T
from collections import namedtuple
import pandas as pd
import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Callable


SaliencyResult = namedtuple('SaliencyResult', ('saliency', 'names', 'weights'))


def get_saliency(
        cost_fn: Callable[('y', 'yhat'), 'scalar'],
        model:T.nn.Module, loader:T.utils.data.DataLoader,
        device:str, num_minibatches:int=float('inf'),
        ):
    """
    Args:
        num_minibatches: Num minibatches from `loader` to get saliency scores.
        cost_fn: reduces y and yhat to a scalar to compute gradient.
            For example: `cost_fn = lambda y,yh: (y*yh).sum()`
    """
    model.to(device, non_blocking=True)
    model.eval()
    # get the set of all 2d spatial filters for all layers of model
    _tmp = list(model.named_parameters())
    weights = [x[1] for x in _tmp]
    names = [x[0] for x in _tmp]
    del _tmp

    # set all filters to requires grad so we can get gradients on them
    [x.requires_grad_(True) for x in weights]

    saliency = [0 for _ in weights]
    N = 0
    for n, (x,y) in enumerate(loader):
        if n >= num_minibatches:
            break
        N += x.shape[0]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = model(x)
        # rescale yhat to all ones or zeros.
        #  (effect of multiplying partial deriv of each class w.r.t. weight times a scalar).
        with T.no_grad():
            yhat /= yhat
            #  yhat[yhat != 0] /= yhat[yhat!=0]

        # get gradients, making all predictions for correct classes correct
        #  (y*yhat).sum().backward(retain_graph=False)
        grads = T.autograd.grad(cost_fn(y, yhat), weights, retain_graph=False)
        with T.no_grad():
            #  for filters, grads in zip(filters_all_layers, grads_all_layers):
            _saliency = [(weights_layer * grads_layer).abs() / N
                        for weights_layer, grads_layer in zip(weights, grads)]
            saliency = [x + y for x,y in zip(saliency, _saliency)]
    return SaliencyResult(
        saliency, names, [x.detach() for x in weights])


def _reinitialize_parameters_(layer:str, layer_inst:T.nn.Module, randvals:T.Tensor):
    """Re-initialize `randvals` in place and return it.  Try to use the default
    initialization available in pytorch."""
    if isinstance(layer_inst, (T.nn.modules.conv._ConvNd)):
        if layer.endswith('.weight'):
            T.nn.init.kaiming_uniform_(randvals)
        elif layer.endswith('.bias'):
            fan_in, _ = T.nn.init._calculate_fan_in_and_fan_out(layer_inst.weight)
            bound = 1 / math.sqrt(fan_in)
            T.nn.init.uniform_(randvals, -bound, bound)
        else:
            raise NotImplementedError()
    elif isinstance(layer_inst, T.nn.modules.batchnorm._NormBase):
        if layer.endswith('.weight'):
            T.nn.init.ones_(randvals)
        elif layer.endswith('.bias'):
            T.nn.init.zeros_(randvals)
        else:
            raise NotImplementedError()
    elif isinstance(layer_inst, T.nn.Linear):
        if layer.endswith('.weight'):
            T.nn.init.kaiming_uniform_(randvals)
        elif layer.endswith('.bias'):
            fan_in, _ = T.nn.init._calculate_fan_in_and_fan_out(layer_inst.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            T.nn.init.uniform_(randvals, -bound, bound)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(f"How to re-initialize layer {layer}")
    return randvals


def get_flat_view(s: SaliencyResult):
    s2s = T.cat([x.view(-1) for x in s.saliency])
    s2w = T.cat([x.view(-1) for x in s.weights])
    s2n = np.array([name for name, w in zip(s.names, s.weights)
                    for _ in range(w.numel())])
    return SaliencyResult(s2s, s2n, s2w)


def reinitialize_least_salient(
        cost_fn: Callable[('y', 'yhat'), 'scalar'],
        model:T.nn.Module, loader:T.utils.data.DataLoader,
        device:str, M:int, frac:float):
    """Re-initialize the `frac` amount of least salient weights in the model with random values.

    `M` minibatch size for saliency
    `frac` fraction of weights with smallest saliency scores in [0,1]
    """
    s = get_saliency(
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
    layer_to_idx = {layer: _idx_bins[v] for v, layer in enumerate(s.names)}
    # todo: take while less than idx_bins in order to group by layer.
    group_by_layer = T.bucketize(s_lo, T.tensor(_idx_bins, device=s_lo.device), right=True)
    # --> Update the (globally) least salient weights across all layers
    for layer in group_by_layer.unique():
        mask = group_by_layer == layer
        flat_idxs = s_lo[mask]
        layer = [sflat.names[i] for i in flat_idxs]
        assert all(x == layer[0] for x in layer)
        layer = layer[0]
        param = model.get_parameter(layer)
        l_idx = layer_to_idx[layer]
        x = np.unravel_index([flat_idxs.cpu().numpy()-l_idx], param.shape)
        # re-initialize weights
        layer_inst = model.get_submodule(layer.rsplit('.', 1)[0])
        randvals = _reinitialize_parameters_(layer, layer_inst, T.empty_like(param.data))
        param.data[x] = randvals[x]
