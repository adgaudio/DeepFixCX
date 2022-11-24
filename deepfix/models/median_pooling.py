"""
Source code copied from
https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, quantile=.5, min_size=None):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same
        self.quantile = quantile
        self.min_size = min_size
        assert quantile == 0.5, 'ensure median pooling'

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        # median pooling
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        # quantile pooling (more gpu memory utilization)
        # x = x.reshape(x.size()[:4] + (-1,)).quantile(self.quantile, dim=-1)
        if self.min_size is not None and any(a<b for a,b in zip(x.shape[-2:], self.min_size)):
                new_size = [max(a,b) for a,b in zip(x.shape, self.min_size)]
                x = torch.nn.functional.interpolate(x, new_size, mode='nearest')
        return x
