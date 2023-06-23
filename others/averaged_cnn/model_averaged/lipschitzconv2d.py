import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

class LipschitzConv2d(_ConvNd):
    def __init__(self, lipschitz, projection, in_channels, out_channels, kernel_size, \
                    signal_size, padding_mode='zeros'):
        """Module for Lipschitz constrained convolution layer"""
        
        padding = _pair(kernel_size // 2)
        kernel_size = _pair(kernel_size)
        stride = _pair(1)
        dilation = _pair(1)
        transposed = False
        output_padding = _pair(0)
        groups = 1
        bias = True
        padding_mode = padding_mode
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, \
                         output_padding, groups, bias, padding_mode)

        self.projection = projection
        self.lipschitz = lipschitz
        # Some projections need largest eigenvector of the layer
        self.additional_parameters = {}
        self.additional_parameters['largest_eigenvector'] = normalize(torch.randn(1, out_channels, signal_size, signal_size))
        self.additional_parameters['end_of_epoch'] = False

        lipschitz_weight, _ = self.projection(self.weight, self.lipschitz, self.additional_parameters)
        self.lipschitz_weight = nn.Parameter(lipschitz_weight)

        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
        else:
            self.padding_mode = padding_mode

    def forward(self, x):
        if self.training:
            lipschitz_weight, new_additional_parameters  = self.projection(self.weight, self.lipschitz, self.additional_parameters)
            self.additional_parameters  = new_additional_parameters
            self.lipschitz_weight.data = lipschitz_weight

            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, self.padding_mode),
                            lipschitz_weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        else:
            
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, self.padding_mode),
                            self.lipschitz_weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

    def set_end_of_training(self):
        self.additional_parameters['end_of_epoch'] = True