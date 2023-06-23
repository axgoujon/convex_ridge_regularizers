import json
import os
import glob
import sys
import math
import numpy as np
import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


from tqdm import tqdm
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


def accelerated_gd(x_noisy, model, ada_restart=False, lmbd=1, mu=1, use_strong_convexity=False, **kwargs):
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
        if use_strong_convexity:
            gamma = (1 - math.sqrt(alpha)) / (1 + math.sqrt(alpha))
        else:
            gamma = (t_old - 1)/t
        z = x + gamma * (x - x_old)

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
                step_size = (2 - 1e-8) / (1 + L*lmbd*mu) 
            else:
                step_size = 2 / (2 + L*lmbd*mu)
            x = x - step_size * ((x - x_noisy) + lmbd * model(mu*x))

    return(x)


def AdaGD(x_noisy, model, lmbd=1, mu=1, **kwargs):
    """denoising with CRR-NNs using the adaptive gradient descent scheme"""

    max_iter = kwargs.get('max_iter', 200)
    tol = kwargs.get('tol', 1e-6)

    def grad_denoising(x):
        return ((x - x_noisy) + lmbd * model(mu*x))

    # initial value
    x_old = torch.clone(x_noisy)

    alpha = 1e-5

    grad = grad_denoising(x_old)
    x = x_old - alpha * grad

    if grad.norm() == 0:
        return(x, 0)

    theta = float('inf')

    for i in range(max_iter):
        grad_old = torch.clone(grad)
        grad = grad_denoising(x)

        alpha_1 = (torch.norm(x - x_old) / (1e-10 + torch.norm(grad - grad_old))).item()/2
        alpha_2 = math.sqrt(1 + theta) * alpha

        alpha_old = alpha
        alpha = min(alpha_1, alpha_2)

        x_old = torch.clone(x)
        
        x = x - alpha * grad

        theta = alpha / (alpha_old + 1e-10)

       
        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if res < tol:
            break
                
    return(x, i)




def AdaGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """solving the inverse problem with CRR-NNs using the adaptive gradient descent scheme"""

    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-6)
    
    enforce_positivity = kwargs.get('enforce_positivity', True)

    def grad_func(x):
        return (Ht(H(x) - y)/op_norm**2+ lmbd * model(mu*x))



    # initial value
    alpha = 1e-5
    x_old = torch.zeros_like(Ht(y))

    grad = grad_func(x_old)

    x = x_old - alpha * grad
    
    pbar = tqdm(range(max_iter))

    theta = float('inf')
    for i in pbar:
        grad_old = torch.clone(grad)
        grad = grad_func(x)

        alpha_1 = (torch.norm(x - x_old) / torch.norm(grad - grad_old)).item()/2
        alpha_2 = math.sqrt(1 + theta) * alpha

        alpha_old = alpha
        alpha = min(alpha_1, alpha_2)

        x_old = torch.clone(x)
        
        x = x - alpha * grad

        if enforce_positivity:
            x = torch.clamp(x, 0, None)

        theta = alpha / alpha_old

       
        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if x_gt is not None:
            psnr_ = psnr(x, x_gt, data_range=1)
            ssim_ = ssim(x, x_gt)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_ = None
            ssim_ = None
            pbar.set_description(f"psnr: {psnr_:.2f} | res: {res:.2e}")

        if res < tol:
            break
                
    return(x, psnr_, ssim_, i)

def AdaAGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """solving the inverse problem with CRR-NNs using the adaptive gradient descent scheme"""

    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-6)
    
    enforce_positivity = kwargs.get('enforce_positivity', True)

    def grad_func(x):
        return (Ht(H(x) - y)/op_norm**2+ lmbd * model(mu*x))



    # initial value
    alpha = 1e-5
    beta = 1e-5
    x_old = torch.zeros_like(Ht(y))

    grad = grad_func(x_old)

    x = x_old - alpha * grad
    z = torch.clone(x)
    
    pbar = tqdm(range(max_iter))

    theta = float('inf')
    Theta = float('inf')
    for i in pbar:
        grad_old = torch.clone(grad)
        grad = grad_func(x)

        alpha_1 = (torch.norm(x - x_old) / torch.norm(grad - grad_old)).item() / 2
        alpha_2 = math.sqrt(1 + theta/2) * alpha

        alpha_old = alpha
        alpha = min(alpha_1, alpha_2)

        beta_1 = 1 / 4 / alpha_1
        beta_2 = math.sqrt(1 + Theta/2) * beta

        beta_old = beta
        beta = min(beta_1, beta_2)

        gamma = (1/math.sqrt(alpha) - math.sqrt(beta)) / (1/math.sqrt(alpha) + math.sqrt(beta)) 

        z_old = torch.clone(z)

        z = x - alpha * grad

        x_old = torch.clone(x)

        x = z + gamma * (z - z_old)
        

        if enforce_positivity:
            x = torch.clamp(x, 0, None)

        theta = alpha / alpha_old
        Theta = beta / beta_old

       
        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if x_gt is not None:
            psnr_ = psnr(x, x_gt, data_range=1)
            ssim_ = ssim(x, x_gt)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_ = None
            ssim_ = None
            pbar.set_description(f"psnr: {psnr_:.2f} | res: {res:.2e}")

        if res < tol:
            break
                
    return(x, psnr_, ssim_, i)