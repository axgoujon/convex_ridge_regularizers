import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import time
import os
import glob

def accelerated_gd_batch(x_noisy, model, sigma=None, ada_restart=False, stop_condition=None, lmbd=1, grad_op=None, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 500)
    tol = kwargs.get('tol', 1e-4)

    # initial value: noisy image
    x = torch.clone(x_noisy)
    z = torch.clone(x_noisy)
    t = torch.ones(x.shape[0], device=x.device).view(-1,1,1,1)

    # cache values of scaling coeff for efficiency
    scaling = model.get_scaling(sigma=sigma)

    # the index of the images that have not converged yet
    idx = torch.arange(0, x.shape[0], device=x.device)
    # relative change in the estimate
    res = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    # mean number of iterations over the batch
    i_mean = 0

    for i in range(max_iter):

        model.scaling = scaling[idx]
        
        x_old = torch.clone(x)

        
        if grad_op is None:

            grad = model.grad_denoising(z[idx], x_noisy[idx], sigma=sigma[idx], cache_wx=False, lmbd=lmbd)
        else:
            grad = grad_op(z[idx])

        
        x[idx] = z[idx] - grad


        t_old = torch.clone(t)
        t = 0.5 * (1 + torch.sqrt(1 + 4*t**2))
        z[idx] = x[idx] + (t_old[idx] - 1)/t[idx] * (x[idx] - x_old[idx])

        if i > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1,2,3)) / (torch.norm(x[idx], p=2, dim=(1,2,3)))

       

        if ada_restart:
            esti = torch.sum(grad*(x[idx] - x_old[idx]), dim=(1,2,3))
            id_restart = (esti > 0).nonzero().view(-1)
            t[idx[id_restart]] = 1
            z[idx[id_restart]] = x[idx[id_restart]]


        condition = (res > tol)
        if stop_condition is None:
            idx = condition.nonzero().view(-1)

        i_mean += torch.sum(condition).item() / x.shape[0]

        if stop_condition is None:
            if torch.max(res) < tol:
                break
        else:

            sct = stop_condition(x, i)

            if sct:
                break


    model.clear_scaling()
    return(x, i, i_mean)
    

def accelerated_gd_single(x_noisy, model, sigma=None, ada_restart=False, stop_condition=None, lmbd=1, grad_op=None, t_init=1, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 500)
    tol = kwargs.get('tol', 1e-4)

    # initial value: noisy image
    x = torch.clone(x_noisy)
    z = torch.clone(x_noisy)
    t = t_init

    model.clear_scaling()
    model.scaling = model.get_scaling(sigma=sigma)
    # the index of the images that have not converged yet
    # relative change in the estimate
    res = 100000


    for i in range(max_iter):

        
        x_old = torch.clone(x)

        if grad_op is None:
            grad = model.grad_denoising(z, x_noisy, sigma=sigma, cache_wx=False, lmbd=lmbd)
        else:
            grad = grad_op(z)

        x = z - grad



        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x + (t_old - 1)/t * (x - x_old)

        if i > 0:
            res = (torch.norm(x - x_old) / (torch.norm(x))).item()

       


        if ada_restart:
            esti = torch.sum(grad*(x[idx] - x_old[idx]), dim=(1,2,3))
            id_restart = (esti > 0).nonzero().view(-1)
            if len(id_restart) > 0:
                print(i, " restart", len(id_restart))
            t[idx[id_restart]] = 1
            z[idx[id_restart]] = x[idx[id_restart]]


        condition = (res > tol)
        if stop_condition is None:
            idx = condition.nonzero().view(-1)



        if stop_condition is None:
            if torch.max(res) < tol:
                break
        else:

            sct = stop_condition(x, i)

            if sct:
                break


        
    model.clear_scaling()
    return(x, i, t)
    

import sys
import os
import glob

sys.path.append('../' + os.path.dirname(__file__))
from wc_conv_net import WCvxConvNet
from pathlib import Path
import json

def load_model(name, device='cuda:0', epoch=None):
    # folder
    current_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    directory = f'{current_directory}/pretrained_models/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    rep = {"[": "[[]", "]": "[]]"}

    name_glob = name.replace("[","tr1u").replace("]","tr2u").replace("tr1u","[[]").replace("tr2u","[]]")

    print(f'{current_directory}/pretrained_models/{name_glob}/checkpoints/*.pth')
    # retrieve last checkpoint
    if epoch is None:
        files = glob.glob(f'{current_directory}/pretrained_models/{name_glob}/checkpoints/*.pth', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]), files)
        epoch = max(epochs)


    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.pth'
    # config file
    config = json.load(open(f'{directory}config.json'.replace("[[]","[").replace("[]]","]")))
   
    # build model

    model, _ = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

    

    model.to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.conv_layer.spectral_norm()
    model.eval()

    return(model)

def build_model(config):
    # ensure consistency of the config file, e.g. number of channels, ranges + enforce constraints

    # 1- Activation function (learnable spline)
    param_spline_activation = config['spline_activation']
    # non expansive increasing splines
    param_spline_activation["slope_min"] = 0
    param_spline_activation["slope_max"] = 1
    # antisymmetric splines
    param_spline_activation["antisymmetric"] = True
    # shared spline
    param_spline_activation["num_activations"] = 1

    # 2- Multi convolution layer
    param_multi_conv = config['multi_convolution']
    if len(param_multi_conv['num_channels']) != (len(param_multi_conv['size_kernels']) + 1):
        raise ValueError("Number of channels specified is not compliant with number of kernel sizes")
    

    param_spline_scaling = config['spline_scaling']
    param_spline_scaling["clamp"] = False
    param_spline_scaling["x_min"] = config['noise_range'][0]
    param_spline_scaling["x_max"] = config['noise_range'][1]
    param_spline_scaling["num_activations"] = config['multi_convolution']['num_channels'][-1]


    model = WCvxConvNet(param_multi_conv=param_multi_conv, param_spline_activation=param_spline_activation, param_spline_scaling=param_spline_scaling, rho_wcvx=config["rho_wcvx"])


    return(model, config)
