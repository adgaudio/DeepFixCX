import torch as T
from collections import namedtuple
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


def costfn_multiclass(yhat:T.Tensor, y:T.Tensor, gain=100):  #, use_sigmoid:bool=False):
    """
    Multi-class cost function for attributing saliency of correct outputs.

    Compute `(yhat * w).sum()*gain` for a weight w that is positive for the
    correct class and negative for incorrect classes, and satisfying `w.sum()
    == 0` so saliency of predictions for correct and incorrect classes are
    equal.

    NOTE: The function doesn't matter much.  Just `yhat.sum()`
    gives nearly the same results.

    Args:
        y: tensor of shape (B, ) containing class indices for each of B samples.
        yhat: tensor of shape (B, C) containing predictions of C classes for
            each of B samples

    Backpropagation pushes the distribution `w = f(y)` through the network.

    `w` is defined as:
        - positive for correct class, negative for incorrect classes
        - constraint: the total sum of pos + neg = 0
        - The vector `w` represents the correct class as +0.5 and remaining C-1
          incorrect classes as -0.5/(C-1) so that we model how well predictions
          agree with picking correct and incorrect classes
    """
    B,C = yhat.shape
    assert y.shape == (B,)
    num_neg_classes = C-1
    num_pos_classes = 1
    w = T.ones_like(yhat) * -1/2 / num_neg_classes * gain
    w[T.arange(B), y] = 1/2 / num_pos_classes * gain
    return (yhat * w).sum()


Y, YHat, Scalar = T.Tensor, T.Tensor, T.Tensor  # for type checking


def get_saliency(
        cost_fn: Callable[['YHat', 'Y'], 'Scalar'],
        model:T.nn.Module, loader:T.utils.data.DataLoader,
        device:str, num_minibatches:int=float('inf'),
    mode:str = 'weight*grad', param_names:Optional[list[str]]=None
):
    """
    Args:
        num_minibatches: Num minibatches from `loader` to get saliency scores.
        cost_fn: reduces y and yhat to a scalar to compute gradient.
            For example: `cost_fn = lambda yh,y: (yh*y).sum()`
        mode: How to compute saliency. One of {'weight*grad', 'grad'}
    """
    model.to(device, non_blocking=True)
    model.eval()
    # get the set of all 2d spatial filters for all layers of model
    if param_names is None:
        _tmp = list(model.named_parameters())
    else:
        _tmp = dict(model.named_parameters())
        _tmp = [(k,_tmp[k]) for k in param_names]
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
        grads = T.autograd.grad(cost_fn(yhat, y), weights, retain_graph=False)
        with T.no_grad():
            #  for filters, grads in zip(filters_all_layers, grads_all_layers):
            if mode == 'weight*grad':
                _saliency = [
                    (weights_layer * grads_layer).abs() / N
                    for weights_layer, grads_layer in zip(weights, grads)]
            elif mode == 'grad':
                _saliency = [(grads_layer).abs() / N for grads_layer in grads]
            elif mode == 'weight':
                _saliency = [(weights_layer).abs() / N for weights_layer in weights]
            else:
                raise NotImplementedError(mode)
            saliency = [x + y for x,y in zip(saliency, _saliency)]
    return SaliencyResult(
        saliency, names, [x.detach() for x in weights])
