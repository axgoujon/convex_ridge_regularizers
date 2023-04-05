import json
import os
import glob
import sys
import math
import torch
sys.path.append('../')
from models.convex_ridge_regularizer import ConvexRidgeRegularizer
from pathlib import Path

def load_model(name, device='cuda:0', epoch=None):
    # folder
    directory = f'{os.path.abspath(os.path.dirname(__file__))}/../trained_models/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    # retrieve last checkpoint if epoch not specified
    if epoch is None:
        files = glob.glob(f'{directory}checkpoints/*.pth', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]), files)
        epoch = max(epochs)
        print(f"--- loading checkpoint from epoch {epoch} ---")

    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.pth'
    # config file
    config = json.load(open(f'{directory}config.json'))
    # build model
    model, _ = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return(model)

def build_model(config):
    # ensure consistentcy
    model = ConvexRidgeRegularizer(kernel_size=config['net_params']['kernel_size'],
                            channels=config['net_params']['channels'],
                            activation_params=config['activation_params'])

    return(model, config)


def accelerated_gd(x_noisy, model, ada_restart=False, lmbd=1, mu=1, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 500)
    tol = kwargs.get('tol', 1e-4)

    # initial value: noisy image
    x = torch.clone(x_noisy)
    z = torch.clone(x_noisy)
    t = 1

    L = model.L

    alpha = 1/( 1 + mu * lmbd * L)

    n_restart = 0

    for i in range(max_iter):

        x_old = torch.clone(x)
        grad_z = alpha * ((z - x_noisy) + lmbd * model(mu * z))
        x = z - grad_z

        # acceleration
        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x + (t_old - 1)/t * (x - x_old)

        if ada_restart:
            if (torch.sum(grad_z*(x - x_old)) > 0):
                t = 1
                z = torch.clone(x_old)
                x = torch.clone(x_old)
                n_restart += 1
                
            else:
                # relative change of norm for terminating
                res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
                if res < tol:
                    break
                
    return(x, i, n_restart)


def tStepDenoiser(model, x_noisy, t_steps=50):
    # learnable reg weight parameter
    lmbd = model.lmbd_transformed
    mu = model.mu_transformed

    # Lipschitz bound of the model
    # estimated in a differentiable way
    if model.training:
        L = torch.clip(model.precise_lipschitz_bound(n_iter=2, differentiable=True), 0.1, None)
        # (we clip L for improved stability for small L
        #  e.g. at initialization L = 0)

        # store it for evalutation mode
        model.L.data = L
    else:
        L = model.L

    x = torch.clone(x_noisy)
    for i in range(t_steps):
        # differentiable steps
        # the stepsize rule can be chosen in two ways:
        # 1. (default) guarantee convergence of the gradient descent scheme => stepsize < 2/L
        # 2. produce averaged t-step denoiser:
        #    - i = 0 (first step) stepsize < 2/L
        #    - i > 0              stepsize = 1/L
        opt = 1
        # nb: the difference in perfomance is very small, and only for small t != 1

        if i == 0:
            # this corresponds to gd step with stepsize 2/L initialization with x_noisy
            # average step and average denoiser for t = 1
            x = x_noisy - 2/(L*mu)*(model(mu*x_noisy))
        else:
            if opt == 1:
                step_size = 2 - 1e-8
            else:
                step_size = 1
            x = x - step_size / (1 + L*lmbd*mu)*((x - x_noisy) + lmbd * model(mu*x))

    return(x)

