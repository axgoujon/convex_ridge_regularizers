import numpy as np
import argparse
import json
import torch
from PIL import Image
import math
import pylops_gpu


class LinOpGrad():
    
    def __init__(self, sizein, device):
        
        self.H = sizein[0]
        self.W = sizein[1]
        self.device = device

        self.Dop_0 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=0, device=device, togpu=(True, True))
        # the default finite difference operator is [0.5, 0, 0-.5], but we want [1, -1, 0]
        # it is the standard finite difference operator and improve the results
        self.Dop_0.Op.h[0,0,0] = 1
        self.Dop_0.Op.h[0,0,1] = -1
        self.Dop_0.Op.h[0,0,2] = 0

        self.Dop_1 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=1, device=device, togpu=(True, True))
        self.Dop_1.Op.h[0,0,0] = 1
        self.Dop_1.Op.h[0,0,1] = -1
        self.Dop_1.Op.h[0,0,2] = 0
    
    
    def apply(self, x):
        x = x.to(torch.float)
        with torch.no_grad():
            x = x.to(self.device)
            out_0 = self.Dop_0*(x.reshape(-1))
            out_1 = self.Dop_1*(x.reshape(-1))
            out = torch.cat([torch.reshape(out_0, (1, self.H, self.W)), torch.reshape(out_1, (1, self.H, self.W))], dim=0)
            #out = out.to(vol.device)
            
        return out
        
        
    def applyJacobianT(self, y):
        
        with torch.no_grad():
            y = y.to(self.device)
            out = torch.reshape(self.Dop_0.H*(y[0,...].view(-1)), (self.H, self.W)) + torch.reshape(self.Dop_1.H*(y[1,...].view(-1)), (self.H, self.W))
            #out = out.to(y.device)
            out = out.unsqueeze(0)
            out = out.unsqueeze(0)
            
        return out # 1 x 1 x K x K


def enforce_box_constraints(x, xmin, xmax):
    out = torch.clamp(x, min=xmin, max=xmax)    
    return out


class CostTV():
    def __init__(self, sizein, lamb, device):
        
        self.sizein = sizein
        self.lamb = lamb
        self.device = device
        #self.bounds = [-float('Inf'), float('Inf')]
        self.bounds = [0.0, float('Inf')]
        
        # Gradient operator
        self.D = LinOpGrad(self.sizein, self.device)
        
        # Parameters for FGP for the prox
        self.gam = 1.0/8
        self.num_iter = 100
        
    
    def apply(self, x):
        """x"""
        with torch.no_grad():
            out = torch.sum(torch.sqrt(torch.sum(torch.pow(self.D.apply(x), 2), dim=0)))
        return self.lamb*out
    
    
    def apply_all(self, x):
        """x"""
        return self.apply(x)
        
        
    def applyProx(self, u, alpha):
        
        with torch.no_grad():
            alpha = alpha*self.lamb
            # Initializations
            P = torch.zeros(2, self.sizein[0], self.sizein[1], device=u.device)
            F = torch.zeros(2, self.sizein[0], self.sizein[1], device=u.device)
            t = 1.0
            for kk in range(self.num_iter):
                #Pnew = F + (self.gam/(alpha))*self.D.apply(u - alpha*self.D.applyJacobianT(F))
                Pnew = F + (self.gam/(alpha))*self.D.apply(enforce_box_constraints(u - alpha*self.D.applyJacobianT(F), self.bounds[0], self.bounds[1]))
                tmp = torch.clamp(torch.sqrt(torch.sum(torch.pow(Pnew, 2), dim=0)), min=1.0)
                Pnew = Pnew/tmp.expand(2, self.sizein[0], self.sizein[1])
                
                tnew = (1 + math.sqrt(1 + 4*(t**2)))/2
                F = Pnew + (t - 1)/tnew*(Pnew - P)
                t = tnew
                P = Pnew
        
        return enforce_box_constraints(u - alpha*self.D.applyJacobianT(P), self.bounds[0], self.bounds[1])


def power_iteration(mask, m, n, num_iter):
    b_k = np.random.rand(m,n)

    for _ in range(num_iter):
        b_kl_forward = np.fft.fft2(b_k, norm='backward') * mask
        A_b_k = np.fft.ifft2(b_kl_forward*mask, norm='forward')  
        A_b_k_norm = np.linalg.norm(A_b_k.reshape(-1))
        b_k = A_b_k/A_b_k_norm

    b_kl_forward = np.fft.fft2(b_k, norm='backward') * mask
    A_b_k = np.fft.ifft2(b_kl_forward*mask, norm='forward')
    alpha = A_b_k[0,0]/b_k[0,0]

    return np.absolute(alpha)


# def scale(img):
#     img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
#     #image = 255 * img
#     image = 1 * img
#     return image


