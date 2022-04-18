import torch as T
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


class QuadTree(T.nn.Module):
    def __init__(self, threshold):
        super().__init()
        self.threshold = threshold
    def forward(self, x):
        return self.quadtree_compress(x, threshold)


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
    def __init__(self,lines,image_width,mlp_activation,threshold):
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
