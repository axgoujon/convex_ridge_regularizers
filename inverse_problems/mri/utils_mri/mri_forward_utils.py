import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.dirname(__file__))


class OpMRI_multicoil_forward(nn.Module):
    
    def __init__(self, mask, smap, device):

        super().__init__()
        self.mask = mask.long().to(device)       # 1 x 1 x H x W
        self.smap = smap.to(device)              # 1 x Nc x H x W
        self.device = device
    
    def forward(self, x):
        # x: 1 x 1 x H x W
        # y: 1 x Nc x H x W
        y = torch.fft.fft2(x*self.smap, norm='ortho')*self.mask
        return y
            

class OpMRI_multicoil_adjoint(nn.Module):
    
    def __init__(self, mask, smap_conj, device):

        super().__init__()
        self.mask = mask.long().to(device)       # 1 x 1 x H x W
        self.smap_conj = smap_conj.to(device)              # 1 x Nc x H x W
        self.device = device
    
    def forward(self, x):
        # x: 1 x Nc x H x W
        # y: 1 x Nc x H x W
        y = torch.sum(torch.real(torch.fft.ifft2(x*self.mask, norm='ortho')*self.smap_conj), dim=1, keepdim=True)
        return y


class OpMRI_singlecoil_forward(nn.Module):
    
    def __init__(self, mask, device):

        super().__init__()
        self.mask = mask.long().to(device)       # 1 x 1 x H x W
        self.device = device
    
    def forward(self, x):
        # x: 1 x 1 x H x W
        # y: 1 x Nc x H x W
        y = torch.fft.fft2(x, norm='ortho')*self.mask
        return y


class OpMRI_singlecoil_adjoint(nn.Module):
    
    def __init__(self, mask, device):

        super().__init__()
        self.mask = mask.long().to(device)       # 1 x 1 x H x W
        self.device = device
    
    def forward(self, x):
        # x: 1 x Nc x H x W
        # y: 1 x Nc x H x W
        y = torch.real(torch.fft.ifft2(x*self.mask, norm='ortho'))
        return y
            

def get_operators(mask, smap, device=None):

    if smap is not None:
        smap_conj = torch.conj(smap)
        fwd_op = OpMRI_multicoil_forward(mask, smap, device)
        adjoint_op = OpMRI_multicoil_adjoint(mask, smap_conj, device)
    else:
        fwd_op = OpMRI_singlecoil_forward(mask, device)
        adjoint_op = OpMRI_singlecoil_adjoint(mask, device)
    return fwd_op, adjoint_op


def get_op_norm(fwd_op, adjoint_op, device, img_size, n_iter=15):
    x = torch.rand((1, 1, img_size[0],img_size[1])).to(device).requires_grad_(False)

    with torch.no_grad():
        for i in range(n_iter):
            x = x / x.norm()
            x = adjoint_op(fwd_op(x))

    return (x.norm().sqrt().item())


def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]
