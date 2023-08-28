import torch
from tqdm import tqdm
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os
sys.path += ['../', '../../', os.path.dirname(__file__)]
from mri.utils_mri.mri_forward_utils import center_crop


def AdaGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """solving the inverse problem with CRR-NNs using the adaptive gradient descent scheme"""

    max_iter = kwargs.get('max_iter', 2000)
    tol = kwargs.get('tol', 1e-6)
    x_init = kwargs.get('x_init', None)
    track_cost = kwargs.get('track_cost', False)
    crop_img = kwargs.get('crop', False)

    enforce_positivity = kwargs.get('enforce_positivity', True)

    def grad_func(x):
        return (Ht(H(x) - y)/op_norm**2 + lmbd * model(mu*x))

    def cost_func(x):
        return(0.5 * torch.norm(H(x) - y)**2 / op_norm**2 + lmbd / mu * (model.cost(mu*x)))


    # initial value
    alpha = 1e-5
    
    if x_init is not None:
        x_old = torch.clone(x_init)
    else:
        x_old = torch.zeros_like(Ht(y))

    if track_cost:
        cost = [cost_func(x_old).item()]
        if crop_img:
            x_crop = center_crop(x_old, [320,320])
            psnr_ = psnr(x_crop, x_gt, data_range=1)
            ssim_ = ssim(x_crop, x_gt)
        else:
            psnr_ = psnr(x_old, x_gt, data_range=1)
            ssim_ = ssim(x_old, x_gt)
        psnr_list = [psnr_.item()]
        ssim_list = [ssim_.item()]
        grad_list = [grad_func(x_old).norm().item()]
        res_list = []


    grad = grad_func(x_old)

    x = x_old - alpha * grad
    
    pbar = tqdm(range(max_iter), dynamic_ncols=True)

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
            if crop_img:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1)
                ssim_ = ssim(x_crop, x_gt)
            else:
                psnr_ = psnr(x, x_gt, data_range=1)
                ssim_ = ssim(x, x_gt)
  
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            pbar.set_description(f"psnr: {psnr_:.2f} | res: {res:.2e}")
            psnr_ = None
            ssim_ = None

        if track_cost:
            cost_ = cost_func(x)
            cost.append(cost_.item())
            psnr_list.append(psnr_.item())
            ssim_list.append(ssim_.item())
            res_list.append(res)
            grad_list.append(grad.norm().item())

        if res < tol:
            break
                
    if track_cost:
        return(x, psnr_list, ssim_list, i, cost, res_list, grad_list)
    else:
        return(x, psnr_, ssim_, i)


def AdaAGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """solving the inverse problem with CRR-NNs using the adaptive gradient descent scheme"""

    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-6)
    x_init = kwargs.get('x_init', None)
    track_cost = kwargs.get('track_cost', False)
    crop_img = kwargs.get('crop', False)
    
    enforce_positivity = kwargs.get('enforce_positivity', True)

    if track_cost:
        cost = []
        psnr_list = []
        ssim_list = []

    def grad_func(x):
        return (Ht(H(x) - y)/op_norm**2+ lmbd * model(mu*x))

    # initial value
    alpha = 1e-5
    beta = 1e-5
    if x_init is not None:
        x_old = torch.clone(x_init)
    else:
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
            if crop_img:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1)
                ssim_ = ssim(x_crop, x_gt)
            else:
                psnr_ = psnr(x, x_gt, data_range=1.0)
                ssim_ = ssim(x, x_gt)

            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_ = None
            ssim_ = None
            pbar.set_description(f"psnr: {psnr_:.2f} | res: {res:.2e}")

        if track_cost:
            cost_ = 0.5 * torch.norm(H(x) - y)**2 / op_norm**2 + lmbd / mu * (model.cost(mu*x))
            cost.append(cost_.item())
            psnr_list.append(psnr_.item())
            ssim_list.append(ssim_.item())

        if res < tol:
            break
                
    if track_cost:
        return(x, psnr_list, ssim_list, i, cost)
    else:
        return(x, psnr_, ssim_, i)
    

def AGD_Recon(y, model, restart=False, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """compute the proximal operator using the FISTA accelerated rule"""

    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-5)
    x_init = kwargs.get('x_init', None)
    track_cost = kwargs.get('track_cost', False)
    enforce_positivity = kwargs.get('enforce_positivity', True)
    strong_convexity_constant = kwargs.get('strong_convexity_constant', 0)
    crop = kwargs.get('crop', False)

    # initial value: noisy image
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = torch.clone(Ht(y)).zero_().detach()

    z = torch.clone(x)
    t = 1

    L = model.L

    alpha = 1/( 1 + mu * lmbd * L)

    n_restart = 0
    def cost_func(x):
        return 0.5 * torch.norm(H(x) - y)**2 / op_norm**2 + lmbd / mu * (model.cost(mu*x))
    
    def grad_func(xx):
        return (Ht(H(xx) - y)/op_norm**2+ lmbd * model(mu*xx))
    
    if track_cost:

        cost = [cost_func(x).item()]
        if crop:
            psnr_list = [psnr(center_crop(x, [320,320]), x_gt, data_range=1.0).item()]
            ssim_list = [ssim(center_crop(x, [320,320]), x_gt).item()]
        else:
            psnr_list = [psnr(x, x_gt, data_range=1.0).item()]
            ssim_list = [ssim(x, x_gt).item()]
        grad_norm_list = [grad_func(x).norm().item()]

   
    
    
    gamma = (1 - math.sqrt(alpha)) / (1 + math.sqrt(alpha))
    pbar = tqdm(range(max_iter))

    for i in pbar:
        x_old = torch.clone(x)
        grad_z = alpha * grad_func(z)


        x = z - grad_z

        if enforce_positivity:
            x = torch.clamp(x, 0, None)

        # acceleration
        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        if strong_convexity_constant > 0:
            gamma = (1 - math.sqrt(alpha * strong_convexity_constant)) / (1 + math.sqrt(alpha * strong_convexity_constant))
        else:
            gamma = (t_old - 1)/t
        z = x + gamma * (x - x_old)

        
       

        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if restart:
            if (torch.sum(grad_z * (x - x_old)) > 0):
                t = 1
                print("restart")
                z = torch.clone(x_old)
                x = torch.clone(x_old)
                n_restart += 1
            elif res < tol:
                break
        elif res < tol:
                break
        
        if x_gt is not None:
            if crop:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1)
                ssim_ = ssim(x_crop, x_gt, data_range=1)
            else:
                psnr_ = psnr(x, x_gt, data_range=1)
                ssim_ = ssim(x, x_gt, data_range=1)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_ = None
            ssim_ = None
            pbar.set_description(f"res: {res:.2e}")

        if track_cost:
            cost_ = cost_func(x)
            cost.append(cost_.item())
            psnr_list.append(psnr_.item())
            ssim_list.append(ssim_.item())
            grad_norm_list.append(torch.norm(grad_z/alpha).item())

    if track_cost:
        return(x, psnr_list, ssim_list, i, cost, n_restart, grad_norm_list)
    else:           
        return(x, psnr_, ssim_, i)


def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]