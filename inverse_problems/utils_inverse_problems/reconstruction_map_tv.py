import torch
from tqdm import tqdm
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os
sys.path += ['../', '../../', os.path.dirname(__file__)]
from mri.utils_mri.mri_forward_utils import center_crop

import sys

sys.path += ['../../others/tv']

from tv_prox import CostTV

def TV_Recon(y, alpha, lmbd, H, Ht, x_gt, **kwargs):


    max_iter = kwargs.get('max_iter', 3000)
    tol = kwargs.get('tol', 1e-6)
    x_init = kwargs.get('x_init', None)

    crop_img = kwargs.get('crop', False)

    if x_init is None:
        x = torch.zeros_like((Ht(y)))
    else:
        x = x_init.clone()

    z = x.clone()
    t = 1
    

    def grad_func(x):
        return alpha * (Ht(H(x) - y))

    cost_tv = CostTV(x.squeeze().shape, lmbd, device=x.device)

    with torch.no_grad():
        pbar = tqdm(range(max_iter), dynamic_ncols=True)
        for i in pbar:
            x_old = torch.clone(x)
            x = z - grad_func(z)

            x = cost_tv.applyProx(x, alpha)

            t_old = t 
            t = 0.5 * (1 + math.sqrt(1 + 4*t**2))

            z = x + (t_old - 1)/t * (x - x_old)

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
                psnr_ = None
                ssim_ = None

            if res < tol:
                break
        
        if crop_img:
            x = center_crop(x, [320,320]) 

        return(x, psnr_, ssim_, i)




