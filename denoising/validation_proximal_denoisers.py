import argparse
import torch
from torch.utils.data import DataLoader

from torchmetrics import StructuralSimilarityIndexMeasure 
from torchmetrics.functional import peak_signal_noise_ratio as psnr
ssim = StructuralSimilarityIndexMeasure()

import os
import sys

sys.path.append("../")
from hyperparameter_tuning.validate_coarse_to_fine import ValidateCoarseToFine
from models import utils
from training.data import dataset


torch.set_num_threads(4)
torch.manual_seed(0)


val_dataset = dataset.H5PY("../training/data/preprocessed/BSD/validation.h5")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# generate one noise for all images => not very clever but remove the randomness and no real impact on best value found
noise = torch.zeros((1, 1, 512, 512)).normal_()

def validate(model, lmbd, mu, sigma=25, tol=1e-6):
    device = model.device
    psnr_val = torch.zeros(len(val_dataloader))
    ssim_val = torch.zeros(len(val_dataloader))
    n_restart_val = torch.zeros(len(val_dataloader))
    n_iter_val = torch.zeros(len(val_dataloader))
    for idx, im in enumerate(val_dataloader):
        im = im.to(device)
        im_noisy = im + sigma/255*noise[:, :, :im.shape[2], :im.shape[3]].to(device)
        im_noisy.requires_grad = False
        # im_denoised, n_iter, n_restart = utils.accelerated_gd(im_noisy, model, lmbd=lmbd, max_iter=2000, tol=tol, mu=mu, use_strong_convexity=True)
        im_denoised, n_iter= utils.AdaGD(im_noisy, model, lmbd=lmbd, max_iter=2000, tol=tol, mu=mu)
        psnr_val[idx] = psnr(im_denoised, im, data_range=1)
        ssim_val[idx] = ssim(im_denoised, im)
        n_iter_val[idx] = n_iter
        n_restart_val[idx] = 0

    return(psnr_val.mean().item(), ssim_val.mean().item(), n_iter_val.mean().item())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cuda", type=str,
                        help='device to use')
    
    args = parser.parse_args()

    device = args.device

    ssim = ssim.to(device)

    for sigma_val in [5]:
        # sigma_train = 5
        sigma_train = 5
        exp_n = f"Sigma_{sigma_train}"

        # here we launch the validation for various t-steps models
        #for t in [1, 2, 5, 10, 20, 30, 50]:
        for t in [10]:
            
            exp_name = f"{exp_n}_t_{t}"
            print(f"**** Validation for model ***** {exp_name}")

            model = utils.load_model(exp_name, device=device)
            model.eval()
            
            model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=False)
            # model.initializeEigen(size=400)
            # L_precise = model.precise_lipschitz_bound(n_iter=200)
            # model.L.data = L_precise

            def score(lmbd, mu):
                with torch.no_grad():
                    return validate(model, lmbd, mu, sigma=sigma_val, tol=5e-5)
            # create directory for the validation if it does not exist
            if not os.path.exists("./validation_scores"):
                os.mkdir("./validation_scores")

            validator = ValidateCoarseToFine(score, dir_name="./validation_scores/", exp_name=f"sigma_val_{sigma_val}_{exp_name.lower()}_ada_nopos", p1_init=1, p2_init=1, p2_max=20, freeze_p2=False, gamma_2_stop=1.1)
            # run the validation
            validator.run()
