import torch
from torch import nn
import math
import sys
sys.path.append("..")
from models.utils import accelerated_gd_batch

class DEQFixedPoint(nn.Module):
    def __init__(self, model, params_fw, params_bw):
        super().__init__()
        self.model = model
        self.solver = anderson
        self.params_fw = params_fw
        self.params_bw = params_bw
        self.f = None
        self.fjvp = None
        
    def forward(self, x_noisy, sigma=None, **kwargs):
        # update spectral norm of the convolutional layer
        if self.model.training:
            self.model.conv_layer.spectral_norm()
            
        # fixed point iteration
        def f(x, x_noisy):
             return(x - self.model.grad_denoising(x, x_noisy, sigma=sigma))

        # jacobian vector product of the fixed point iteration
        def fjvp(x, y):
            with torch.no_grad():
                return(y - self.model.hvp_denoising(x, y, sigma=sigma))

        self.f = f
        self.fjvp = fjvp


        # compute the fixed point
        with torch.no_grad():
            z, self.forward_niter_max, self.forward_niter_mean = accelerated_gd_batch(x_noisy, self.model, sigma=sigma, ada_restart=True, **self.params_fw)
      
        z = self.f(z, x_noisy)
  
        z0 = z.clone().detach()

        def backward_hook(grad):
            g, self.backward_niter = self.solver(lambda y : self.fjvp(z0, y) + grad,
                                           grad, **self.params_bw)
            return g

        if self.model.training:
            z.register_hook(backward_hook)
       
        return z



def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
  
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:,1:n+1,0]
   
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break

    return X[:,k%m].view_as(x0), k


