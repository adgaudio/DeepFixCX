import torch as T
import numpy as np
from typing import Union
import cv2
import torch.nn as nn


class HLine(T.nn.Module):
    def __init__(self,lines):
        super().__init__()
        self.lines=lines

    def forward(self, x: T.Tensor):
        """
        Args:
            x:  a minibatch of x-ray images of shape (B,1,H,W)
        """
        hlines = x[:,:,self.lines, :]
        return hlines


class RLine(T.nn.Module):
    def __init__(self, img_HW, nlines=25,
                 initialization='inscribed_circle', seed=None):
        if initialization == 'inscribed_circle':
            #  seed 0
            center_HW = img_HW[0]/2, img_HW[1]/2
            radius = min(img_HW[0], img_HW[1])/2
            # For each line, randomly choose two (x,y) endpoint coords
            # by sampling uniformly from the perimeter of a circle
            # --> generate random points in N(0,1)
            x = np.random.RandomState(seed).randn(nlines, 2, 2)
            # --> ensure that the endpoints are on opposite half-circles
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
            arr = np.zeros(img_HW, dtype='float32')
            [cv2.line(arr, tuple(p1), tuple(p2), 1) for p1,p2 in x]
        else:
            raise NotImplementedError()
        self.arr = T.tensor(arr)


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


class QTHlineClassifier(T.nn.Module):
    def __init__(self,lines:Union[HLine,RLine],image_width,mlp_activation,threshold):
        """
        Args:
            lines: list of row indices in img.
            image_width: number of columns in image.
            threshold:  value of variance to choose
        """
        super().__init__()

        self.hline=HLine(lines=lines)

        if threshold is not None:
            self.quadtree=QuadTree(threshold)
        else:
            self.quadtree=None

        mlp_input_size=len(lines)*image_width
        self.mlp = MlpClassifier(activation=mlp_activation,input_size=mlp_input_size)

    def forward(self, x):
        if self.quadtree is not None:
            x = self.quadtree(x)
        x = self.hline(x)
        return self.mlp(x)
