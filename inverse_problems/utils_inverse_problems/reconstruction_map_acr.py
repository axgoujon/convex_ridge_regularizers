import numpy as np
from tqdm import tqdm
import torch
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


def GD_Recon_torch(y, model, lmbd=1, H=None, Ht=None, x_gt=None, **kwargs):
    """solving the inverse problem with ACR using GD"""
    # No convergence guarantees => have to set the max_iter cautiously

    max_iter = kwargs.get('max_iter', 400)
    alpha = kwargs.get('alpha', 0.8)

    x_init = kwargs.get('x_init', None)

    if x_init is not None:
        x = x_init.clone().to(y.device).clamp(0, 1).requires_grad_(False)
       
    else:
        x = torch.clone(Ht(y)).zero_().to(y.device).requires_grad_(True)

    # sq_loss = torch.nn.MSELoss(reduction='mean')

    # optim = torch.optim.SGD([x], lr=alpha)
  
    pbar = tqdm(range(max_iter), dynamic_ncols=True)

    for i in pbar:
        x_old = torch.clone(x)

        #optim.zero_grad()

        #Hx = H(x)

        #data_loss = sq_loss(Hx, y)

        #reg_loss = lmbd * model(x)

        #total_loss = reg_loss + data_loss

        #total_loss.backward(retain_graph=True)
        #optim.step()
        with torch.no_grad():
            g1 = Ht(H(x) - y)
        g2 = model.grad(x)    

        x = x - alpha * (0.5 * g1 + lmbd * g2)
  
        # relative change of norm for terminating
        if i > 0:
            res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
       

            if x_gt is not None:
                psnr_ = psnr(x, x_gt, data_range=1.0)
                ssim_ = ssim(x, x_gt)
                pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.3f} | res: {res:.2e}")
            else:
                pbar.set_description(f"res: {res:.2e}")
                psnr_ = None
                ssim_ = None

            # tolerance cannot be used because no convergence guarantees
                
    return(x, psnr_, ssim_, i)
