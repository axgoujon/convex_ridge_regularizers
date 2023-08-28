import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import sys
import os

sys.path.append(os.path.dirname(__file__))
from spline_module import LinearSpline, clip_activation
from multi_conv import MultiConv2d
import time

class WCvxConvNet(nn.Module):
    def __init__(self, param_multi_conv, param_spline_activation, param_spline_scaling, rho_wcvx=1):
        
        super().__init__()

        self.num_channels = param_multi_conv["num_channels"][-1]
        # 1 - multi-convolutionnal layers
        print("Multi convolutionnal layer: ", param_multi_conv)
        self.conv_layer = MultiConv2d(**param_multi_conv)

        # 2 - activation functions (gradient of the potential)
        # a - increasing part, gradient of the convex part of the potential
        param_spline_activation["init"] = "zero"
        param_spline_activation["slope_min"] = 0
        param_spline_activation["slope_max"] = 1

        self.activation_cvx = LinearSpline(**param_spline_activation)
        # b - decreasing part, gradient of the concave part of the potential
        # different initialization
        param_spline_activation_ccv = param_spline_activation.copy()
        param_spline_activation_ccv["init"] = "identity"
        param_spline_activation_ccv["slope_min"] = 0
        param_spline_activation_ccv["slope_max"] = 1
        self.activation_ccv = LinearSpline(**param_spline_activation_ccv)
        
        # 3 - mu parameter (controls the magnitude of the convex part of the potential / increasing part of spline)
        self.mu_ = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        
        # 4 - scaling parameter to add some flexibility to the activation accross channels and noise levels
        self.spline_scaling = LinearSpline(**param_spline_scaling)
        
        self.num_params = sum(p.numel() for p in self.parameters())

        # to cache the scaling and mu
        self.scaling = None

        # cached values
        self.cached_wx = None
        # set to 0 if cvx, to 1 if 1 weakly convex, etc
        self.rho_wcvx = rho_wcvx



    @property
    def device(self):
        return(self.conv_layer.conv_layers[0].weight.device)
        
    def get_scaling(self, sigma=None):
        if self.scaling is None:
            eps = 1e-5
            return(torch.exp(self.spline_scaling(torch.tile(sigma, (1, self.num_channels, 1, 1)))) / (sigma + eps))
        else:
            return(self.scaling)
        
    def cache_scaling(self, sigma=None):
        self.scaling = self.get_scaling(sigma)

    def clear_scaling(self):
        self.scaling = None


    def get_mu(self):
        return(self.mu_.exp())

    def activation(self, x, sigma=None, skip_scaling=False):
        # get scaling, which depends on sigma and on the channel
        if not skip_scaling:
            scaling = self.get_scaling(sigma)
        else:
            scaling = 1
        x = x * scaling
        # apply activation
        y = self.get_mu() * self.activation_cvx(x) - self.rho_wcvx * self.activation_ccv(x)
        # scale back
        y = y / scaling

        return(y)

    def grad_activation(self, x, sigma=None):
        scaling = self.get_scaling(sigma)
        x = x * scaling
        return(self.get_mu() * self.activation_cvx.derivative(x) - self.rho_wcvx * self.activation_ccv.derivative(x))


    def integrate_activation(self, x, sigma=None, skip_scaling=False):

        if not skip_scaling:
            scaling = self.get_scaling(sigma)
        else:
            scaling = 1

        x = x * scaling

        x_ccv = self.activation_ccv.integrate(x)

        y = self.get_mu() * self.activation_cvx.integrate(x) - self.rho_wcvx * x_ccv

        y = y / scaling / scaling

        return(y)

    def grad(self, x, sigma=None, cache_wx=False):

        """ Gradient of the loss at location x. Update conv.L before if needed."""
        # first multi convolution layer
        y = self.conv_layer(x)

        if cache_wx:
            self.cached_wx = y

        # activation
        y = self.activation(y, sigma=sigma)

        y =  self.conv_layer.transpose(y)
 
        return(y)

    def grad_denoising(self, x, x_noisy, sigma=None, cache_wx=False, lmbd=1):

        return(1/(1 + lmbd*self.get_mu()) * ((x - x_noisy) + lmbd*self.grad(x, sigma=sigma, cache_wx=cache_wx)))

    def hvp(self, x, v, sigma=None):

        """ Hessian of R vector product """
        # first multi convolution layer on x and v
        y_x = self.conv_layer(x)
        y_v = self.conv_layer(v)

        # derivative activation at y_v
        y_x_1 = self.grad_activation(y_x, sigma=sigma)

        y = y_x_1 * y_v

        y =  self.conv_layer.transpose(y)

        return(y)

    def hvp_denoising(self, x, v, sigma=None):

        return(1/(1 + self.get_mu()) * (v + self.hvp(x, v, sigma=sigma)))


    def update_integrated_params(self):
        self.activation_cvx.hyper_param_to_device()
        self.activation_ccv.hyper_param_to_device()
        self.activation_cvx.update_integrated_coeff()
        self.activation_ccv.update_integrated_coeff()

    def cost(self, x, sigma, use_cached_wx=False):
        s = x.shape
        # first multi convolution layer
        
        if use_cached_wx:
            y = self.cached_wx
        else:
            y = self.conv_layer(x)
        #print(y.shape)
        # activation
        y = self.integrate_activation(y, sigma)
        #print(y.shape)

        return(torch.sum(y, dim=tuple(range(1, len(s)))))

    def change_splines_to_clip(self):
        # for inference only
        # splines decomposed in clip functions for faster computation
        # still under development so not very robust!
        activation_cvx = self.activation_cvx
        activation_ccv = self.activation_ccv

        activation_cvx.hyper_param_to_device()
        activation_ccv.hyper_param_to_device()
        
        self.activation_cvx = self.activation_cvx.get_clip_equivalent()
        

        grid_tensor = torch.linspace(activation_cvx.x_min.item(), activation_cvx.x_max.item(), activation_cvx.num_knots, device=self.device).expand((activation_cvx.num_activations, activation_cvx.num_knots))

        coeff_proj = activation_ccv.projected_coefficients.clone().to(self.device)
        i0 = torch.arange(0, coeff_proj.shape[0]).to(coeff_proj.device)
        x1 = grid_tensor[i0, 1].view(1,-1,1,1)
        y1 = coeff_proj[i0, 1].view(1,-1,1,1)

        x2 = grid_tensor[i0, -2].view(1,-1,1,1)
        y2 = coeff_proj[i0, -2].view(1,-1,1,1)

        slopes = ((y2 - y1)/(x2 - x1)).view(1,-1,1,1)

        self.activation_ccv = clip_activation(x1, x2, y1, slopes)

        mid = grid_tensor.shape[1] // 2
        x1 = grid_tensor[i0, mid].view(1,-1,1,1)
        y1 = coeff_proj[i0, mid + 1].view(1,-1,1,1)

        x2 = grid_tensor[i0, mid + 1].view(1,-1,1,1)
        y2 = coeff_proj[i0, mid + 1].view(1,-1,1,1)

        slopes_0 = ((y2 - y1)/(x2 - x1)).view(1,-1,1,1)

        self.activation_cvx.slopes += slopes_0 / self.get_mu()





