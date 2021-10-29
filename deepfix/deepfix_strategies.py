from typing import Callable
import dataclasses as dc
import torch as T
from simplepytorch import trainlib as TL
from deepfix.weight_saliency import costfn_multiclass
from deepfix.init_from_distribution import init_from_hist_, reinitialize_least_salient
from deepfix.models.ghaarconv import GHaarConv2d


class DeepFix_TrainOneEpoch:
    """
    DeepFix: Re-initialize bottom `P` fraction of least salient weights
    every `N` epochs.  Wraps a `train_one_epoch` function.

    Example Usage:
        >>> using_deepfix = DeepFixTrainingStrategy(
                N=3, P=.2, train_one_epoch_fn=TL.train_one_epoch)
        >>> cfg = TrainConfig(..., train_one_epoch=using_deepfix)
        >>> cfg.train()
    """
    def __init__(self, N:int, P:float, R:int,
                 train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result]):
        """
        Args:
            N: Re-initialize weights every N epochs.  N>1.
            P: The fraction of least salient weights to re-initialize.  0<=P<=1.
            R: Num repetitions.  How many times to re-initialize in a row.
            train_one_epoch_fn: The function used to train the model.
                Expects as input a TL.TrainConfig and outputs a TL.Result.
        """
        assert 0 <= P <= 1, 'sanity check: 0 <= P <= 1'
        assert N >= 1, 'sanity check: N > 1'
        assert R >=1
        self.N, self.P, self.R = N, P, R
        self._wrapped_train_one_epoch = train_one_epoch_fn
        self._counter = 0

    def __call__(self, cfg:TL.TrainConfig):
        self._counter = (self._counter + 1) % self.N
        if self._counter % self.N == 0:
            print('DeepFix REINIT')
            for i in range(self.R):
                reinitialize_least_salient(
                    costfn_multiclass,
                    #  lambda y, yh: yh[y].sum(),  # TODO: wrong
                    model=cfg.model, loader=cfg.train_loader,
                    device=cfg.device, M=50, frac=self.P*((self.R-i)/self.R),
                    opt=cfg.optimizer
                )
        return self._wrapped_train_one_epoch(cfg)


@dc.dataclass
class DeepFix_LambdaInit:
    init_fn: Callable[T.nn.Module,None]
    args:tuple = ()
    train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result] = TL.train_one_epoch
    _called = False

    def __call__(self, cfg: TL.TrainConfig):
        if not self._called:
            self._called = True
            print(f"DeepFix {self.init_fn.__name__}{self.args}")
            self.init_fn(cfg.model, *self.args)
        return self.train_one_epoch_fn(cfg)


@dc.dataclass
class DeepFix_GHaarConv2d:
    train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result] = TL.train_one_epoch
    kwargs:dict = dc.field(default_factory=dict)

    def __call__(self, cfg: TL.TrainConfig):
        print('Replace Spatial Conv2d layers with GHaarConv2d')
        DeepFix_GHaarConv2d.convert_conv2d_to_gHaarConv2d(cfg.model, self.kwargs)
        return self.train_one_epoch_fn(cfg)

    @staticmethod
    def convert_conv2d_to_gHaarConv2d(model: T.nn.Module, kwargs):
        """Replace all spatial Conv2d layers with GHaarConv2d(**kwargs),
        where kwargs overrides any defaults already defined in conv2d layer.
        Idea adapted from https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736/8

        Note: GHaarConv2d doesn't support padding_mode.
        """
        recurse_on_these = []
        for attr_name, conv2d in model.named_children():
            if not isinstance(conv2d, T.nn.Conv2d):
                recurse_on_these.append(conv2d)
                continue
            if conv2d.kernel_size[0] <= 1 or conv2d.kernel_size[1] <= 1:
                continue
            kws = dict(
                in_channels=conv2d.in_channels,
                out_channels=conv2d.out_channels,
                kernel_size=conv2d.kernel_size, stride=conv2d.stride,
                padding=conv2d.padding, dilation=conv2d.dilation,
                groups=conv2d.groups, bias=conv2d.bias)
            kws.update(kwargs)
            new_conv2d = GHaarConv2d(**kws).to(conv2d.weight.device)
            from efficientnet_pytorch.utils import Conv2dStaticSamePadding
            if isinstance(conv2d, Conv2dStaticSamePadding):
                # workaround for efficientnet
                new_conv2d = T.nn.Sequential(
                    conv2d.static_padding,
                    new_conv2d)
            elif issubclass(conv2d.__class__, T.nn.Conv2d) and conv2d.__class__ != T.nn.Conv2d:
                print(
                    f"WARNING: converted an instance of {conv2d.__class__}that inherits from conv2d to"
                    " a GHaarConv2d.  This might cause bugs.")
            setattr(model, attr_name, new_conv2d)
        # --> recursive through child modules.
        for child_module in recurse_on_these:
            DeepFix_GHaarConv2d.convert_conv2d_to_gHaarConv2d(child_module, kwargs)
        return model

@dc.dataclass
class DeepFix_DHist:
    fp: str
    train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result] = TL.train_one_epoch
    init_with_hist:bool = True
    fixed:bool = False
    _called = False

    def __call__(self, cfg: TL.TrainConfig):
        if not self._called:
            self._called = True
            if self.init_with_hist:
                hist = T.load(self.fp)
                print('DeepFix HISTOGRAM Initialization')
                init_from_hist_(cfg.model, hist)
            if self.fixed:
                for layer in cfg.model.modules():
                    layer.requires_grad_(False)
                for name,x in list(cfg.model.named_modules())[::-1]:
                    if len(list(x.parameters())):
                        x.requires_grad_(True)
                        break
                print(f"DeepFix:  all layers fixed ...except layer: {name}")
        return self.train_one_epoch_fn(cfg)
