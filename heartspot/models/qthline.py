import torch as T
from matplotlib import pyplot as plt
import numpy as np
from typing import Union, List
import cv2
import torch.nn as nn
from heartspot.models.quadtree import QT
from heartspot.models.median_pooling import MedianPool2d
from heartspot.models import get_densenet


class HLine(T.nn.Module):
    def __init__(self, lines:List[int], width:int):
        super().__init__()
        self.lines=lines
        self.width = width
        self.out_size = len(lines) * width

    def forward(self, x: T.Tensor):
        """
        Args:
            x:  a minibatch of x-ray images of shape (B,1,H,W)
        """
        W = x.shape[-1]
        center_crop = (W-self.width)//2
        hlines = x[:,:,self.lines, center_crop:self.width+center_crop]
        return hlines


class RLine(T.nn.Module):
    def __init__(self, img_HW, nlines=25, zero_top_frac:int=0,
                 heart_roi:bool=False,
                 initialization='inscribed_circle', seed=None,
                 hlines:List[int]=(), sum_aggregate=False, ret_img=False):
        super().__init__()
        self.sum_aggregate = sum_aggregate
        self.ret_img = ret_img
        assert sum_aggregate + ret_img <= 1
        if initialization == 'inscribed_circle':
            #  seed 0
            center_HW = img_HW[0]/2, img_HW[1]/2
            #  radius = min(img_HW[0], img_HW[1])/2
            radius = max(img_HW[0], img_HW[1])
            # For each line, randomly choose two (x,y) endpoint coords
            # by sampling uniformly from the perimeter of a circle
            # --> generate random points in N(0,1)
            x = np.random.RandomState(seed).randn(nlines, 2, 2)
            # --> ensure that the endpoints are on opposite half-circles,
            # putting points are either on left or right half-circle.
            x[:,0,0] = np.abs(x[:,0,0])
            x[:,1,0] = -1.*np.abs(x[:,1,0])
            #  x[:,0,1] = np.abs(x[:,0,1])
            #  x[:,1,1] = -1.*np.abs(x[:,1,1])
            # --> project points onto the perimeter of circle
            x = x / ((x**2).sum(-1, keepdims=True)**.5)
            # --> inscribe the circle into the image
            x = x * radius + center_HW
            # --> convert to pixel coordinates
            x = np.round(x).astype('int')
            # Generate the lines using a line algorithm from cv2
            arr = np.zeros(img_HW, dtype='int8')
            [cv2.line(arr, tuple(p1), tuple(p2), 1) for p1,p2 in x]
        else:
            raise NotImplementedError()
        if zero_top_frac:
            arr[:int(zero_top_frac * arr.shape[0])] = 0
        if hlines:
            arr[hlines] = 1
        if heart_roi:
            # load image
            roi = T.tensor(plt.imread('data/cardiomegaly_mask.jpg'))
            # resize image.
            roi = T.nn.functional.interpolate(
                roi.unsqueeze_(0).unsqueeze_(0), size=img_HW,
                mode='nearest').squeeze()
            if arr.sum() == 0:  # there must be no lines.  include the inner region.
                arr[roi >= 100] = 1
            else:  # zero out the lines outside the region.
                arr[roi < 100] = 0
        if arr.sum() == 0:
            arr = np.ones_like(arr)
        if self.sum_aggregate:
            self.out_size = arr.shape[-2] + arr.shape[-1]
        else:
            self.out_size = arr.sum()
        self.arr = T.tensor(arr, dtype=T.bool).unsqueeze_(0).unsqueeze_(0)

    def forward(self, x):
        assert x.shape[-2:] == self.arr.shape[-2:], f'expected spatial dims {self.arr.shape[-2:]} but got {x.shape[-2:]}'
        if self.sum_aggregate:
            z1 = x.sum(-2), x.sum(-1)
            x[(~self.arr).repeat(x.shape[0], x.shape[1],1,1)] = 0
            z2 = x.sum(-2), x.sum(-1)
            return T.cat([x.sum(-2), x.sum(-1)], -1)
        elif self.ret_img:
            x[~(self.arr.repeat(x.shape[0], x.shape[1],1,1))] = 0
            return x
        else:
            B, C = x.shape[:2]
            x = x[self.arr.repeat(x.shape[0], x.shape[1],1,1)]
            return x.reshape(B, C, -1)


class MedianPoolDenseNet(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = T.nn.Sequential(
            MedianPool2d(kernel_size=12, stride=4),
            get_densenet('densenet121', 'untrained', 1, 1))
    def forward(self, x):
        return self.fn(x)


class MlpClassifier(T.nn.Module):
    def __init__(self, input_size, activation_fn=T.nn.SELU):
        super().__init__()
        self.mlp = T.nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,3200),
            nn.BatchNorm1d(3200),
            activation_fn(),
            nn.Linear(3200,300),
            nn.BatchNorm1d(300),
            activation_fn(),
            nn.Linear(300,1),
        )
    def forward(self,x):
        op = self.mlp(x)
        return op


class QTLineClassifier(T.nn.Module):
    def __init__(self,lines:Union[HLine,RLine], quadtree:Union[T.nn.Module,QT]):
        """
        Args:
            lines: a pytorch module that extracts lines or points from data.
            threshold:  value of variance to choose
        """
        super().__init__()
        self.lines_fn = lines
        self.quadtree = quadtree
        self.mlp = MlpClassifier(input_size=lines.out_size)

    def forward(self, x):
        with T.no_grad():
            if self.quadtree is not None:
                x = self.quadtree(x)
            x = self.lines_fn(x)
        return self.mlp(x)
