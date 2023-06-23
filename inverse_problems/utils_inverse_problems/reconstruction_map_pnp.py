import torch
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os
sys.path += ['../', '../../', os.path.dirname(__file__)]
from mri.utils_mri.mri_forward_utils import center_crop
import math

def PnP_Recon_FBS(y, model, alpha, lmbd, H, Ht, x_gt, **kwargs):

    max_iter = kwargs.get('max_iter', 3000)
    tol = kwargs.get('tol', 5e-6)
    mode = kwargs.get('mode', 'residual')
    x_init = kwargs.get('x_init', None)

    crop_img = kwargs.get('crop', False)

    if x_init is not None:
        x = x_init
    else:
        x = torch.zeros_like((Ht(y)))

    
    def grad_func(x):
        return (Ht(H(x) - y))

    with torch.no_grad():
        pbar = tqdm(range(max_iter), dynamic_ncols=True)
        for i in pbar:
            
            x_old = torch.clone(x)
            grad = grad_func(x)
            x = x - alpha * grad

            # relaxed denoising step
            # to control the strength of denoising, we use a convex combination of the denoised image and the previous estimate
            if mode == 'residual':
                x_denoised = x - model(x)
            else:
                x_denoised = 0.5*(model(x) + x)
            x = lmbd * x_denoised + (1 - lmbd) * x
            
            # relative change of norm for terminating
            res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

            if x_gt is not None:
                if crop_img:
                    x_crop = center_crop(x, [320,320])
                    psnr_ = psnr(x_crop, x_gt, data_range=1.0)
                    ssim_ = ssim(x_crop, x_gt, data_range=1.0)
                    pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.3f} | res: {res:.2e}")
                
                else:
                    psnr_ = psnr(x, x_gt, data_range=1.0)
                    ssim_ = ssim(x, x_gt, data_range=1.0)
                    pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.3f} | res: {res:.2e}")
            
            else:
                pbar.set_description(f"res: {res:.2e}")

            if res < tol:
                break
        
        if crop_img:
            x = center_crop(x, [320,320])   

        return(x, psnr_, ssim_, i)
    

def PnP_Recon_FISTA(y, model, alpha, lmbd, H, Ht, x_gt, **kwargs):

    max_iter = kwargs.get('max_iter', 3000)
    tol = kwargs.get('tol', 5e-6)
    mode = kwargs.get('mode', 'residual')
    x_init = kwargs.get('x_init', None)

    crop_img = kwargs.get('crop', False)

    if x_init is not None:
        x = x_init
    else:
        x = torch.zeros_like((Ht(y)))

    x0 = x.clone()
    tk = 1.0
    
    def grad_func(x):
        return (Ht(H(x) - y))

    with torch.no_grad():
        pbar = tqdm(range(max_iter), dynamic_ncols=True)
        for i in pbar:
            
            x_old = torch.clone(x)
            grad = grad_func(x0)
            x = torch.clamp(x0 - alpha * grad, 0.0, None)

            # relaxed denoising step
            # to control the strength of denoising, we use a convex combination of the denoised image and the previous estimate
            if mode == 'residual':
                x_denoised = x - model(x)
            else:
                x_denoised = 0.5*(model(x) + x)
            x = lmbd * x_denoised + (1 - lmbd) * x

            told = tk
            tk = 0.5*(1 + math.sqrt(1 + 4*(tk**2)))
            x0 = x + 1.0*((told-1)/tk)*(x - x_old)
            
            # relative change of norm for terminating
            res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

            if x_gt is not None:
                if crop_img:
                    x_crop = center_crop(x, [320,320])
                    psnr_ = psnr(x_crop, x_gt, data_range=1.0)
                    ssim_ = ssim(x_crop, x_gt, data_range=1.0)
                    pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.3f} | res: {res:.2e}")
                
                else:
                    psnr_ = psnr(x, x_gt, data_range=1.0)
                    ssim_ = ssim(x, x_gt, data_range=1.0)
                    pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.3f} | res: {res:.2e}")
            
            else:
                pbar.set_description(f"res: {res:.2e}")

            if res < tol:
                break
            
        return(x, psnr_, ssim_, i)