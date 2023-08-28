import torch
from tqdm import tqdm
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os
sys.path += ['../', '../../', os.path.dirname(__file__)]
from mri.utils_mri.mri_forward_utils import center_crop


def SAGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """ """
    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-5)
    x_init = kwargs.get('x_init', None)
    enforce_positivity = kwargs.get('enforce_positivity', True)
    crop = kwargs.get('crop', False)

    # initial value: noisy image
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = torch.clone(Ht(y)).zero_().detach()

    z = torch.clone(x)
    x_old = torch.clone(x)
    z_old = torch.clone(x)
    t = 1
    t_old = 1

    
    sigma = torch.ones((1, 1, 1, 1), device=x.device) * mu

    model.mu = None
    model.scaling = None

    scaling = model.get_scaling(sigma=sigma)

    mu = model.get_mu()
    model.mu = mu
    model.scaling = scaling

    restart_count = 0



    L = (model.mu * lmbd + op_norm**2)

    
    def grad_func(xx):
        return (Ht(H(xx) - y)+ lmbd * model.grad(xx, sigma))
  
    beta = 1.001
    
    pbar = tqdm(range(max_iter))


    for i in pbar:
        z = x + (t_old - 1)/t * (x - x_old)
        
        grad_z = grad_func(z)


        crit = (torch.sum((grad_z*(z - z_old))) + 1/2 * lmbd * beta * torch.sum((z - z_old)**2)).item()

        if crit > 0:
            z = x.clone()
            t = 1
            t_old = 1
            restart_count += 1
            grad_z = grad_func(z)

        x_old = torch.clone(x)
        
        x = z - 1/L * grad_z
        if enforce_positivity:
            x = torch.clamp(x, 0, None)
        t_old = t
        t = (1 + math.sqrt(1 + 4 * t**2))/2

        z_old = torch.clone(z)

       

        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if res < tol:
                break
        
        if x_gt is not None:
            if crop:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1)
                ssim_ = ssim(x_crop, x_gt, data_range=1)
            else:
                psnr_ = psnr(x, x_gt, data_range=1.0)
                ssim_ = ssim(x, x_gt, data_range=1.0)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_ = None
            ssim_ = None
            pbar.set_description(f"res: {res:.2e}")


    return(x, psnr_, ssim_, i)
