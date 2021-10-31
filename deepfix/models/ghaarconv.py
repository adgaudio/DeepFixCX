import torch as T
import math
from typing import Union, Tuple


class GHaarConv2d(T.nn.Module):
    """
    Learnable Generalized Haar Wavelets Filters.

    Drop-in replacement for Conv2d (supporting the most common keyword
    arguments) that uses the generalized Haar filters.  The filter size can
    have any shape 2x2 or larger, and regardless of size, a spatial filter is
    always created from 4 parameters: h, v, d, f.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        #  padding_mode: str = 'zeros',  # not implemented
        constrain_f: bool = False,
    ):
        super().__init__()
        assert isinstance(kernel_size, (int, tuple))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.hvdf = T.nn.Parameter(T.Tensor(out_channels, in_channels // groups, 4))
        if bias is not None:
            self.bias = T.nn.Parameter(T.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.conv2d_hyperparams = dict(
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=self.bias)
        self.reset_parameters()
        self.filters = None  # filters are cached in eval mode.
        self.constrain_f = constrain_f

    def reset_parameters(self) -> None:
        # init h, v, d weights
        T.nn.init.uniform_(self.hvdf[..., :3], -1, 1)
        # init frequency
        T.nn.init.uniform_(self.hvdf[..., 3], 0, 2*(T.mean(self.kernel_size)-1))
        # bias, if enabled
        if self.bias is not None:  # kaiming uniform initialization
            in_ch = self.hvdf.shape[1]
            #  out_ch = self.hvdf.shape[0]
            receptive_field_size = math.prod(self.kernel_size)
            fan_in = in_ch * receptive_field_size
            #  fan_out = out_ch * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            T.nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def ghaar(kernel_size, hvdf, bigger=0, constrain_f=False):
        assert hvdf.shape[-1] == 4, hvdf.shape
        y_bigger = math.pi/(kernel_size[0]-1)*(bigger//2)
        x_bigger = math.pi/(kernel_size[1]-1)*(bigger//2)
        y = T.linspace(0-y_bigger, math.pi+y_bigger, kernel_size[0]+bigger,
                       device=hvdf.device).reshape(1,1,kernel_size[0]+bigger,1)
        x = T.linspace(0-x_bigger, math.pi+x_bigger, kernel_size[1]+bigger,
                       device=hvdf.device).reshape(1,1,1,kernel_size[1]+bigger)
        h = hvdf[..., 0].unsqueeze(-1).unsqueeze(-1)
        v = hvdf[..., 1].unsqueeze(-1).unsqueeze(-1)
        d = hvdf[..., 2].unsqueeze(-1).unsqueeze(-1)
        f = hvdf[..., 3].unsqueeze(-1).unsqueeze(-1)
        if constrain_f:
            #  f_y = f % ((kernel_size[0]-1) *2)
            #  f_x = f % ((kernel_size[1]-1) *2)
            f_y = T.sigmoid(f)*(kernel_size[0]-1)*2
            f_x = T.sigmoid(f)*(kernel_size[1]-1)*2
        else:
            f_x, f_y = f, f
        t = math.pi/2
        x_h = T.sin(x*f_x+t) ; y_h = T.sin(y*0+t)
        x_v = T.sin(x*0+t) ; y_v = T.sin(y*f_y+t)
        x_d = T.sin(x*f_x+t) ; y_d = T.sin(y*f_y+t)
        filter = h*(x_h*y_h) + v*(x_v*y_v) + d*(x_d*y_d)
        return filter

    def forward(self, x):
        # cache the filters during model.eval(). NO cache during model.train()
        if not self.training:
            if self.filters is None:
                filters = self.filters = self.ghaar(self.kernel_size, self.hvdf)
            else:
                filters = self.filters
        else:
            self.filters = None
            filters = self.ghaar(self.kernel_size, self.hvdf, constrain_f=self.constrain_f)
        return T.conv2d(x, filters, **self.conv2d_hyperparams)


def convert_conv2d_to_gHaarConv2d(
        model: T.nn.Module, ignore_layers:list[str] = (), kwargs:dict=None):
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
        if attr_name in ignore_layers:
            print("SKIP", attr_name)
            continue
        kws = dict(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size, stride=conv2d.stride,
            padding=conv2d.padding, dilation=conv2d.dilation,
            groups=conv2d.groups, bias=conv2d.bias)
        kws.update(kwargs or {})
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
        convert_conv2d_to_gHaarConv2d(child_module, ignore_layers, kwargs)
    return model
