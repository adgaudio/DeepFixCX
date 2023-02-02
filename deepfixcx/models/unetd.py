import torch as T
from typing import Tuple, Union, Any
from pampy import match
            #  levels (depth)
            #  channels (width)
            #  filter size
            #  stride
            #  dilation

            #  downsampling


def compose_fns(fns, x):
    """Apply function composition with a series of functions on given input  f2(f1(x)).
    Return output of each f_i(x)"""
    outputs = [x]
    for i, fn in enumerate(fns):
        try:
            outputs.append(fn(outputs[-1]))
        except Exception as e:
            try:
                shape = outputs[-1].shape
            except:
                try:
                    shape = len(outputs[-1])
                except:
                    shape = 'unknown'
            msg = f"ERROR at step {i}, evaluating fn {fn.__class__.__name__} on {type(x)} of size {shape}"
            print(msg)
            raise
    return outputs

def compose_reduce2_fns(fns, iterable, start_val):
    """Apply reduce operation on a series of functions and series of inputs.
    The operation is  f_i(x_{i-1}, x_i).
    Return outputs of each f_i(x_{i-1}, x_i)
    """
    prev_x = [start_val]
    for fn, next_x in zip(fns, iterable):
        prev_x.append(fn(prev_x[-1], next_x))
    return prev_x


class Block_DepthwiseSeparableConv2d(T.nn.Module):
    """
    Perform a (1x1 conv -> Z -> NxN grouped conv -> Z -> 1x1 conv -> Z)
        where Z is (batchnorm -> relu)

        with options to configure Z, N, num channels, dilations, strides, padding, etc.

        Based on MobileNetV2 paper.
    """
    #  def search_space_cfg(self):
    #      ch_in, ch_out = self.convs[0].channels_in, self.convs[-1].channels_out
    #      return dict(
    #          channels_mid=tune.randint(ch_in, ch_out*10),
    #          kernel_size=tune.randint(0,8),

    #      )
    #  def search_space_parse_cfg(self, cfg):
    #      cfg = dict(cfg)  # no longer mutable keys
    #      cfg['kernel_size'] = cfg['kernel_size'] * 2 + 1
    #      cfg

    #      return dct
    def __init__(self, channels_in:int, channels_mid:int, channels_out:int,
                 kernel_size:int = 3,
                 paddings:Tuple[Union[str,int]] = (0,1,0),
                 dilations:Tuple[int] = (1,1,1),
                 strides:Tuple[int] = (1,1,1),
                 activations:Tuple[T.nn.Module] = (T.nn.CELU(inplace=True),T.nn.CELU(inplace=True),T.nn.Identity()),
                 batch_norms:Tuple[Union[bool, T.nn.Module]] = (True,True,True)):
        super().__init__()
        # --> prepare input data a little
        conv_kws = [
            dict(padding=paddings[i], dilation=dilations[i], stride=strides[i])
            for i in (0,1,2)]
        batch_norms = tuple(
            match(bn,
                  True, lambda x: T.nn.BatchNorm2d(ch),
                  False, lambda x: T.nn.Identity(),
                  Any, bn)
            for bn, ch in zip(batch_norms, (channels_mid, channels_mid, channels_out)))
        # --> define the convolutions
        convs = [
            T.nn.Conv2d(channels_in, channels_mid, kernel_size=1, bias=False, **conv_kws[0]),
            batch_norms[0],
            activations[0],
            T.nn.Conv2d(channels_mid, channels_mid, kernel_size=3, groups=channels_mid, bias=False, **conv_kws[1]),
            batch_norms[1],
            activations[1],
            T.nn.Conv2d(channels_mid, channels_out, kernel_size=1, bias=False, **conv_kws[2]),
            batch_norms[2],
            activations[2],
        ]
        self.convs = T.nn.ModuleList(convs)

    def forward(self, x):
        return compose_fns(self.convs, x)[-1]


class Downsample(T.nn.Sequential):
    def __init__(self, ch_in, ch_mid, ch_out, kernel_size):
        super().__init__(
            Block_DepthwiseSeparableConv2d(
                ch_in, ch_mid, ch_out, kernel_size=kernel_size,
                paddings=(0,1,0), strides=(1,1,1)),
            Block_DepthwiseSeparableConv2d(  # DOWNSAMPLE HERE
                ch_out, ch_mid, ch_out, kernel_size=kernel_size,
                paddings=(0,1,0), strides=(1,kernel_size,1))
        )


class Upsample(T.nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, kernel_size):
        super().__init__()
        self.layers_pre_fusion = T.nn.Sequential(
            # (implicity applied in forward): UPSAMPLE (via interpolation)
            Block_DepthwiseSeparableConv2d(
                ch_in, ch_mid, ch_out, kernel_size=kernel_size,
                paddings=(0,1,0), strides=(1,1,1)),
        )
        self.layers_post_fusion = T.nn.Sequential(
            Block_DepthwiseSeparableConv2d(
                ch_out, ch_mid, ch_out, kernel_size=kernel_size,
                paddings=(0,1,0), strides=(1,1,1))
        )
        self.w = T.nn.Parameter(T.ones(2, dtype=T.float)/2)

    def forward(self, x_prev, x_next):
        x_prev_U = T.nn.functional.interpolate(
            x_prev, size=x_next.shape[-2:], mode='nearest')
        x_prev_U = self.layers_pre_fusion(x_prev_U)

        # compute a fusion image by weighted sum, based on EfficientDet paper
        w1, w2 = self.w[0], self.w[1]
        Z = w1 + w2 + 1e-5
        #  fusion = T.cat([w1/Z*x_prev_U, w2/Z*x_next], 1)
        fusion = w1/Z*(x_prev_U.contiguous()) + w2/Z*(x_next.contiguous())

        ret = self.layers_post_fusion(fusion)
        return ret


class UnetD(T.nn.Module):
    """U-Net architecture that by default:
      - variable number of layers and channels per layer
      - uses only depthwise separable convolutions, based on MobileNetV2,
        with default configurable option to increase channels by 6x before each grouped conv
      - instead of ReLU activations, use CELU
      - in decoder, instead of concatenate, use a sum fusion layer
      - in decoder, use interpolation and depthwise separable convolution
        rather than transpose convolution.
    """

    def __init__(self, channels: Tuple[int]=(3,32,64,128,256,512,1024,),
                 depthwise_channel_multiplier: int=12,
                 head: T.nn.Module=None, tail: T.nn.Module=None):
        """
        searchables(i,j,k):
            channels = i, 12i, j
            kernel_size = k
            activations = (T.nn.CELU(inplace=True),T.nn.CELU(inplace=True),T.nn.Identity()),
            batch_norms =(True,True,True)):

        fixed
            paddings:Tuple[Union[str,int]] = (0,(k-1)/2,0)
            dilations:Tuple[int] = (1,1,1),
            strides:Tuple[int] = (1,1,1),
        """
        super().__init__()
        enc, dec = [], []
        for i in range(len(channels)-1):
            kws = dict(
                ch_in=channels[i],
                ch_mid=depthwise_channel_multiplier*channels[i],
                ch_out=channels[i+1],
                kernel_size=3)
            enc.append(Downsample(**kws))
            kws['ch_in'], kws['ch_out'] = kws['ch_out'], kws['ch_in']
            dec.append(Upsample(**kws))  # ideally, appendLeft

        self.head = head if head is not None else T.nn.Sequential()
        self.layers_encoder = T.nn.ModuleList(enc)
        self.layers_decoder = T.nn.ModuleList(dec[::-1])
        self.tail = tail if tail is not None else T.nn.Sequential()

    def forward_encoder(self, x):
        return compose_fns(self.layers_encoder, x)

    def forward_decoder(self, encoder_activations):
        return compose_reduce2_fns(
            self.layers_decoder, encoder_activations[-2::-1], encoder_activations[-1])

    def forward(self, x):
        output_head = self.head(x)
        output_encoder = self.forward_encoder(output_head)
        output_decoder = self.forward_decoder(output_encoder)
        output_tail = self.tail(output_decoder[-1])
        return output_tail
