import torch
from torch import nn
import torch.nn.utils.parametrize as P

import math
from math import sqrt

import numpy as np

class MultiConv2d(nn.Module):
    """A module to perform multi-convolutions, i.e. composition of convolutions
        This was found helpful at training, especailly to be able to
        train convolutional layers with large kernel size
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        """
        Parameters
        ----------
        channels : list of int
            The list of channels from the input to output
            e.g. [1, 8, 32] for 2 convolutional layers with 1 input channels, 32 output channels and 8 channels in between
        kernel_size : int
            The size of the kernel of all convolutional layers
        padding : int
            The padding should be kernel_size // 2 to prserve the spatial size of the input
        """
        super().__init__()
        # parameters
        self.padding = padding
        self.kernel_size = kernel_size
        self.channels = channels

        # convolutionnal layers
        self.conv_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size, padding=self.padding, bias=False))
            P.register_parametrization(self.conv_layers[-1], "weight", ZeroMean())
       
        # scale the filters to ensure spectral norm of one at init
        self.initSN()

      

    def forward(self, x):
        return(self.convolution(x))
    
    def convolution(self, x):
        for conv in self.conv_layers:
            x = nn.functional.conv2d(x, conv.weight, padding=self.padding, dilation=conv.dilation)
        return(x)


    def transpose(self, x):
        """Transpose of the linear operation.
         For CRR-NNs this should be handled carefully to ensure
         that the output is the actual transpose. Otherwise convexity of reg can be lost"""
        for conv in reversed(self.conv_layers):
            # there are two variants to implement the transpose of a conv with 0 padding in torch
            # x = nn.functional.conv2d(x.flip(2,3), conv.weight.permute(1, 0, 2, 3), padding=self.padding).flip(2, 3)
            weight = conv.weight
            x = nn.functional.conv_transpose2d(x, weight, padding=conv.padding, groups=conv.groups, dilation=conv.dilation)

        return(x)
    
    def spectral_norm(self, n_power_iterations=10, size=40):
        u = torch.empty((1, self.conv_layers[0].weight.shape[1], size, size), device=self.conv_layers[0].weight.device).normal_()
        with torch.no_grad():
            for _ in range(n_power_iterations):
                # In this loop we estimate the eigen vector u corresponding to the largest eigne value of W^T W
                v = normalize(self.convolution(u))
                u = normalize(self.transpose(v))
                if n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()
            cur_sigma = torch.sum(u * self.transpose(v))

            return cur_sigma


    def initSN(self):
        with torch.no_grad():
            cur_sn = self.spectral_norm()
            for conv in self.conv_layers:
                conv.weight.data = conv.weight.data/(cur_sn**(1/len(self.conv_layers)))

    def checkTranpose(self):
        """Check if the transpose is correctly implemented"""
        x = torch.randn((1, 1, 40, 40), device=self.conv_layers[0].weight.device)
        Hx = self.forward(x)
        y = torch.randn((1, Hx.shape[1], 40, 40), device=self.conv_layers[0].weight.device)

        Hty = self.transpose(y)


        v1 = torch.sum(Hx * y)
        v2 = torch.sum(x * Hty)

        print("tranpose check", torch.max(torch.abs(v1 - v2))) 

    


def normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor**2)))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans

# zero mean kernel parametrization
class ZeroMean(nn.Module):
    def forward(self, X):
        return(X - torch.mean(X, dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))

                