import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(__file__))

from lipschitzconv2d import LipschitzConv2d
from conv_projections import spectral_norm_conv
from nonexpansive_prelu import LipschitzPReLU



class AveragedCNN(nn.Module):
    """CNN for averaged denoisers"""
    def __init__(self, num_channels=64, num_layers=9, kernel_size=3, signal_size=256):
        super().__init__()

        modules = nn.ModuleList()

        projection = spectral_norm_conv
        

        lipschitz_bound = 1


        # First block
        modules.append(LipschitzConv2d(lipschitz_bound, projection, 1, num_channels, kernel_size, signal_size))
        modules.append(LipschitzPReLU(num_parameters=num_channels))

        # Middle blocks
        for i in range(num_layers-2):
            modules.append(LipschitzConv2d(lipschitz_bound, projection, num_channels, num_channels, kernel_size, signal_size))
            modules.append(LipschitzPReLU(num_parameters=num_channels))

        # Last block
        modules.append(LipschitzConv2d(lipschitz_bound, projection, num_channels, 1, kernel_size, signal_size))

        self.layers = nn.Sequential(*modules)

        self.num_params = sum(p.numel() for p in self.parameters())
        

    def forward(self, x):
        """ """
        return self.layers(x)
    
