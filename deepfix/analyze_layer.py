from dataclasses import dataclass
from typing import Optional, Callable
import torch as T


@dataclass(frozen=True, repr=False)
class DataPoint:
    """Assemble data associated to a module, a specific input and gradient.

    The repr of this function will use `metadata['id']` if it exists to
    identify this DataPoint.
    """
    module_name: str  # i.e. from model.named_modules()
    metadata: Optional[dict]
    # e.g. metadata can identify which dataset samples, gradient cost function,
    # and model was used for this data point.
    #
    input: T.Tensor
    grad_input: T.Tensor
    #
    output: T.Tensor
    grad_output: T.Tensor

    def __repr__(self):
        id = self.metadata.get('id')
        id = '' if id is None else f', id={id}'
        return f'{self.__class__.__name__}({self.module_name}{id})'


@dataclass(frozen=True, repr=False)
class LayerDataPoint(DataPoint):
    weight: T.Tensor
    grad_weight: T.Tensor
    #
    bias: Optional[T.Tensor] = None
    grad_bias: Optional[T.Tensor] = None


@dataclass(frozen=True, repr=False)
class ModuleDataPoint(DataPoint):
    params: dict[str, T.Tensor]
    grad_params: dict[str, T.Tensor]

    def __post_init__(self):
        assert isinstance(self.params, dict)
        assert isinstance(self.grad_params, dict)
        assert set(self.params) == set(self.grad_params)


Y = YHat = T.tensor  # for type checking


def _analyze_model_at_modules__setup(model, module_names):
    """Create the forward hooks to capture activations, and get the flat list
    of parameters to get gradients of weights with a way to assemble the
    parameters back to their layers"""
    # initialize pre- and post- activation hooks to capture forward pass
    pre_acts, post_acts = [], []
    all_named_params = []
    hooks = []
    for module_name in module_names:
        module = model.get_submodule(module_name)
        def fhook(mod, input, output):
            pre_acts.append(input)
            post_acts.append(output)
        hooks.append(module.register_forward_hook(fhook))
        all_named_params.append((module_name, list(module.named_parameters())))
    # assemble flat list of params to obtain gradients
    params, lookup_param_to_flatidx, _i = [], {}, 0
    for _mod_name, _lst in all_named_params:
        for _param_name, _param in _lst:
            params.append(_param)
            lookup_param_to_flatidx[(_mod_name, _param_name)] = _i
            _i += 1
    del _mod_name, _lst, _param_name, _param, _i
    return pre_acts, post_acts, hooks, all_named_params, params, lookup_param_to_flatidx


def analyze_model_at_modules(
    model:T.nn.Module, module_names:list[str], loader:T.utils.data.DataLoader,
    grad_cost_fn: Callable[['YHat', 'Y'], T.Tensor], device:str
) -> dict[str, list[DataPoint]]:
    """Extract data for detailed data associated to chosen layers.
    Uses a lot of RAM if passing multiple `module_names` or multiple samples
    in the `loader`.

    Args:
        model: any pytorch model
        module_names: list of the modules in `model` that you wish to analyze.
            e.g. [k for k,v in module.named_modules() if isinstance(k, T.nn.Conv2d)]
            e.g. ['conv1', 'layer1.0.conv1', ...]
        loader: pytorch DataLoader.  May want `batch_size=1, shuffle=False` if
            analyzing gradients of single inputs/images.  Data loader should
            generate pairs (X,y) where X is model input, and y is ground truth.
        grad_cost_fn: a function that reduces the model output `YHat` and given
            ground truth `Y` to a scalar value in order to compute gradient.
        device: pytorch device such as 'cpu' or 'cuda:0'
    Return:
        A dict of {module_name: [DataPoint, ...]}
          containing the pre/post-activations, weights and all gradients
          associated to that module.  The DataPoint is either a LayerDataPoint
          or a more general ModuleDataPoint.  The LayerDataPoint is used if we can
          assume each layer has a .weight and optional .bias as parameters.
          The ModuleDataPoint is used otherwise (enabling representation of
          arbitrary modules)
    """
    model = model.eval().to(device, non_blocking=True)
    (pre_acts, post_acts, hooks, all_named_params, params_flat, lookup_param_to_flatidx) = \
        _analyze_model_at_modules__setup(model, module_names)
    # collect data with forward and backward passes
    returned_results = {mn: [] for mn in module_names}
    for i, (module_name, named_params) in enumerate(all_named_params):
        for jth_minibatch, (X, y) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            del pre_acts[:], post_acts[:]
            X.requires_grad_(True)
            yhat = model(X)
            loss = grad_cost_fn(yhat, y)
            grads_w = T.autograd.grad(loss, params_flat, retain_graph=True)
            grads_pre = T.autograd.grad(loss, [x for y in pre_acts for x in y], retain_graph=True)
            grads_post = T.autograd.grad(loss, post_acts, retain_graph=False)
            assert len(grads_w) == len(params_flat) >= len(all_named_params)
            assert len(grads_pre) == len(pre_acts)
            assert len(grads_post) == len(post_acts)
            assert len(all_named_params) == len(pre_acts) == len(post_acts)
            # assemble all weights associated to a given module (layer)
            with T.no_grad():
                kws1 = dict(
                    module_name=module_name,
                    metadata={'id': jth_minibatch, 'X': X.detach().cpu(),
                              'y': y.detach().cpu(), 'yhat': yhat.detach().cpu()},
                    input=[x.detach().cpu() for x in pre_acts[i]],
                    grad_input=[x.detach().cpu() for x in grads_pre[i]],
                    output=post_acts[i].detach().cpu(),
                    grad_output=grads_post[i].detach().cpu(),
                )
                tmp = dict(named_params)
                if len(tmp) <= 2 and not set(tmp).difference({'weight', 'bias'}):
                    dp = LayerDataPoint(
                        weight=tmp['weight'].detach().cpu(),
                        grad_weight=grads_w[lookup_param_to_flatidx[(module_name, 'weight')]].detach().cpu(),
                        bias=tmp['bias'].detach().cpu() if 'bias' in tmp else None,
                        grad_bias=grads_w[lookup_param_to_flatidx[(module_name, 'bias')]].detach().cpu() if tmp.get('bias') is not None else None,
                        **kws1, 
                    )
                else:
                    dp = ModuleDataPoint(
                        params=tmp,
                        grad_params={k: grads_w[lookup_param_to_flatidx[(module_name, k)]].detach().cpu() for k in tmp},
                        **kws1,)
                returned_results[module_name].append(dp)
    for h in hooks:
        h.remove()
    return returned_results
