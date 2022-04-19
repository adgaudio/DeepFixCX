import torch as T
import numpy as np
from typing import Union, List
import cv2
import torch.nn as nn


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
    def __init__(self, img_HW, nlines=25,
                 initialization='inscribed_circle', seed=None):
        super().__init__()
        if initialization == 'inscribed_circle':
            #  seed 0
            center_HW = img_HW[0]/2, img_HW[1]/2
            radius = min(img_HW[0], img_HW[1])/2
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
        self.out_size = arr.sum()
        self.arr = T.tensor(arr, dtype=T.bool).unsqueeze_(0).unsqueeze_(0)

    def forward(self, x):
        assert x.shape[-2:] == self.arr.shape[-2:]
        return x[self.arr.repeat(x.shape[0], x.shape[1],1,1)]\
                .reshape(x.shape[0], x.shape[1], -1)


class QuadTree(T.nn.Module):
    def __init__(self, threshold):
        super().__init()
        self.threshold = threshold

    def forward(self, x):
        return self.quadtree_compress(x, self.threshold)


class MlpClassifier(T.nn.Module):
    def __init__(self, activation, input_size):
        super().__init__()

        if activation == 'CELU':
            activation_fn = nn.CELU()
        elif activation == 'SELU':
            activation_fn = nn.SELU()
        elif activation == 'Tanh':
            activation_fn = nn.Tanh()
        elif activation == 'RELU':
            activation_fn = nn.ReLU()

        layers = [nn.Flatten()]
        layers += [nn.Linear(input_size,300)]
        layers += [nn.BatchNorm1d(300)]
        layers += [activation_fn]
        layers += [nn.Linear(300,1)]
        layers += [nn.Sigmoid()]
        self.mlp = T.nn.Sequential(*layers)

    def forward(self,x):
        op = self.mlp(x)
        return op


class QTLineClassifier(T.nn.Module):
    def __init__(self,lines:Union[HLine,RLine], mlp_activation, threshold):
        """
        Args:
            lines: a pytorch module that extracts lines or points from data.
            threshold:  value of variance to choose
        """
        super().__init__()

        self.lines_fn=lines

        if threshold is not None:
            self.quadtree=QuadTree(threshold)
        else:
            self.quadtree=None

        self.mlp = MlpClassifier(
            activation=mlp_activation, input_size=lines.out_size)

    def forward(self, x):
        if self.quadtree is not None:
            x = self.quadtree(x)
        x = self.lines_fn(x)
        return self.mlp(x)
