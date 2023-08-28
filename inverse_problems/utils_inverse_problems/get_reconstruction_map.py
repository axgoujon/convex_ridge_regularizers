import sys
import argparse
import torch
import os

sys.path += ['../', '../../', os.path.dirname(__file__)]




def get_reconstruction_map(method, modality, device="cuda:0", **opts):
    if modality == 'ct':
        from ct.utils_ct.ct_forward_utils import get_operators as get_operators_ct
        from ct.utils_ct.ct_forward_utils import get_op_norm as get_op_norm_ct

        H, fbp, Ht = get_operators_ct(device=device)
        op_norm = get_op_norm_ct(H, Ht, device=device)

    elif modality == 'mri':
        from mri.utils_mri.mri_forward_utils import get_operators as get_operators_mri
        from mri.utils_mri.mri_forward_utils import get_op_norm as get_op_norm_mri

    tol = opts.get('tol', 1e-5)
    if method == 'crr':
        from utils_inverse_problems.reconstruction_map_crr import AdaGD_Recon, AGD_Recon
        from models.utils import load_model
        
        model_name = opts.get('model_name', None)

        if model_name is None:
            raise ValueError("Please provide a model_name for the crr model. It is the name of the folder corrsponding to the trained model.")
        
        model = load_model(model_name, epoch=10, device=device)
        model.eval()
        model.prune(prune_filters=True, collapse_filters=False, change_splines_to_clip=False)

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    algo_name = opts.get('algo_name', 'adagd')
                    if algo_name == 'adagd':
                        x, psnr_, ssim_, n_iter = AdaGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    elif algo_name == 'fista':
                        x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    else:
                        raise NotImplementedError
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
    elif method == 'wcrr':
        from utils_inverse_problems.reconstruction_map_wcrr import SAGD_Recon
        sys.path += ['../../others/wcrr/model_wcrr/']
        from utils import load_model as load_model_wcrr
        
        
        model_name = opts.get('model_name', 'WCRR-CNN')

        if model_name is None:
            raise ValueError("Please provide a model_name for the wcrr model. It is the name of the folder corrsponding to the trained model.")
        
        model = load_model_wcrr(model_name, device=device)
        model.eval()

        sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=1000)


        if modality == 'ct':

            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    x, psnr_, ssim_, n_iter = SAGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = SAGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
        
    elif method == 'tv':
        if modality == 'ct':
            from utils_inverse_problems.reconstruction_map_tv import TV_Recon
            def reconstruction_map(y, p1, x_gt=None, x_init=None, **kwargs):
                with torch.no_grad():
                    x, psnr_, ssim_, n_iter = TV_Recon(y, alpha=1/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, **opts)
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            from utils_inverse_problems.reconstruction_map_tv import TV_Recon
            def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None, **kwargs):
                with torch.no_grad():
                    H, Ht = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = TV_Recon(y, alpha=1/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
        
    elif method == 'pnp':
        from utils_inverse_problems.reconstruction_map_pnp import PnP_Recon_FBS, PnP_Recon_FISTA

        # load model
        model_type = opts.get('model_type', None)
        model_sigma = opts.get('model_sigma', None)
        if model_type is None or model_sigma is None:
            raise ValueError("Please provide a model_type and a model_sigma for the pnp model.")
        
        if model_type == "dncnn":
            sys.path += ['../../others/dncnn/model_dncnn/']
            from utils_dncnn import load_model as load_model_dncnn

            model = load_model_dncnn(model_sigma, device=device)
            model.eval()
            mode = "residual"

        elif model_type == "averaged_cnn":
            sys.path += ['../../others/dncnn/model_dncnn/', '../../others/averaged_cnn/model_averaged/']
            from utils_averaged_cnn import load_model as load_model_averaged_cnn

            mode = "direct"
            model = load_model_averaged_cnn(model_sigma, device=device)
            model.eval()

        else:
            raise ValueError(f"model_type {model_type} not supported")
        
        if modality == 'ct':
            if opts.get('n_hyperparameters', 1) == 2:
                def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                    with torch.no_grad():
                        x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=p2/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_init, **opts)
                    return(x, psnr_, ssim_, n_iter)
            else:
                def reconstruction_map(y, p1, x_gt=None, x_init=None):
                    with torch.no_grad():
                        x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=1.99/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_init, **opts)
                    return(x, psnr_, ssim_, n_iter)

        elif modality == 'mri':
            if opts.get('n_hyperparameters', 1) == 2:

                if model_type == "dncnn":

                    def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FISTA(y, model, alpha=p2/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)
                    
                elif model_type == "averaged_cnn":

                    def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=p2/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)

            else:

                if model_type == "dncnn":

                    def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FISTA(y, model, alpha=1.0/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)
                    
                elif model_type == "averaged_cnn":

                    def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=1.99/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)

        else:
            raise NotImplementedError

  
    elif method == 'acr':
        from utils_inverse_problems.reconstruction_map_acr import GD_Recon_torch
        sys.path += ['../../others/acr/model_acr/']
        from utils_acr import load_model as load_model_acr


        model = load_model_acr(device=device)
        model.eval()

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                x, psnr_, ssim_, n_iter = GD_Recon_torch(y, model, lmbd=p1, alpha=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, **opts)
                return(x, psnr_, ssim_, n_iter)
            
        elif modality == 'mri':
            raise NotImplementedError
        
        else:
            raise NotImplementedError
       
            
    else:
        raise NotImplementedError
    
    return(reconstruction_map)
