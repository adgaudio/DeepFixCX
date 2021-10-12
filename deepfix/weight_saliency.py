import torch as T
from collections import namedtuple
import math
import numpy as np
from typing import Callable, Optional
import dataclasses as dc


@dc.dataclass
class SaliencyResult:
    """Output of a get_saliency(...) call, containing the saliency scores for
    each parameter in the model.
    A parameter is, for example, `layer.weight` or `layer.bias`.
    All parameters are output of `model.named_parameters()`.

    Example Usage:
        ```
        sr = SaliencyResult(saliency=..., param_names=..., weights=...)
        for psr in sr:
            print(psr)
            break
        ```
    """
    saliency: list[T.Tensor]
    param_names: list[str]
    weights: list[T.Tensor]

    ParamSaliencyResult = namedtuple(
        'ParamSaliencyResult', ['saliency', 'param_name', 'weight'])

    def __post_init__(self):
        assert len(self.saliency) == len(self.param_names) == len(self.weights), 'sanity check: matching sizes'

    def __iter__(self):
        for a,b,c in zip(self.saliency, self.param_names, self.weights):
            yield self.ParamSaliencyResult(a,b,c)


def costfn_multiclass(y:T.Tensor, yhat:T.Tensor, gain=30):  #, use_sigmoid:bool=False):
    """
    Multi-class cost function for weight saliency of correct outputs.

    Compute `(yhat * w).sum()` and the gradient w.r.t `yhat` is `w`.
    Just ignore model predictions `yhat`.

    NOTE: The function doesn't matter much.  Just `yhat.sum()` gives nearly the
    same results.

    Args:
        y: tensor of shape (B, ) containing class indices for each of B samples.
        yhat: tensor of shape (B, C) containing predictions of C classes for
            each of B samples

    Backpropagion pushes the distribution `w = f(y)` through the network.

    `w` is defined as:
        - positive for correct class, negative for incorrect classes
        - constraint: the total sum of pos + neg = 0
    """
    B,C = yhat.shape
    assert y.shape == (B,)
    num_neg_classes = C-1
    num_pos_classes = 1
    w = T.ones_like(yhat) * -1/2 / num_neg_classes
    w[T.arange(B), y] = 1/2 / num_pos_classes
    return (yhat * w).sum() * 30


def get_saliency(
        cost_fn: Callable[('y', 'yhat'), 'scalar'],
        model:T.nn.Module, loader:T.utils.data.DataLoader,
        device:str, num_minibatches:int=float('inf'),
        mode:str = 'weight*grad'
):
    """
    Args:
        num_minibatches: Num minibatches from `loader` to get saliency scores.
        cost_fn: reduces y and yhat to a scalar to compute gradient.
            For example: `cost_fn = lambda y,yh: (y*yh).sum()`
        mode: How to compute saliency. One of {'weight*grad', 'grad'}
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
        #  with T.no_grad():
            #  yhat /= yhat
            #  yhat[yhat != 0] /= yhat[yhat!=0]

        # get gradients, making all predictions for correct classes correct
        #  (y*yhat).sum().backward(retain_graph=False)
        grads = T.autograd.grad(cost_fn(y, yhat), weights, retain_graph=False)
        with T.no_grad():
            #  for filters, grads in zip(filters_all_layers, grads_all_layers):
            if mode == 'weight*grad':
                _saliency = [
                    (weights_layer * grads_layer).abs() / N
                    for weights_layer, grads_layer in zip(weights, grads)]
            elif mode == 'grad':
                _saliency = [(grads_layer).abs() / N for grads_layer in grads]
            else:
                raise NotImplementedError(mode)
                #  _saliency = [(weights_layer).abs() / N
                #              for weights_layer in weights]
            saliency = [x + y for x,y in zip(saliency, _saliency)]
    return SaliencyResult(
        saliency, names, [x.detach() for x in weights])


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


def get_flat_view(s: SaliencyResult):
    s2s = T.cat([x.view(-1) for x in s.saliency])
    s2w = T.cat([x.view(-1) for x in s.weights])
    s2n = np.array([name for name, w in zip(s.param_names, s.weights)
                    for _ in range(w.numel())])
    return SaliencyResult(s2s, s2n, s2w)


def reinitialize_least_salient(
        cost_fn: Callable[('y', 'yhat'), 'scalar'],
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
                assert len(z) == 1, f"TODO: optimizer has other parameters might need to re-initialize: {z.keys()}"
            else:
                assert len(z) == 0, f'TODO:  optimizer has other parameters might need to re-initialize: {z.keys()}'


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
