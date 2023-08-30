import torch
import os
import sys
import pandas as pd
import numpy as np

sys.path.append('../../')
sys.path.append(os.path.dirname(__file__))

from get_reconstruction_map import get_reconstruction_map
from hyperparameter_tuning.validate_coarse_to_fine import ValidateCoarseToFine
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


def validate(method, modality, device, job_name, **kwargs):

    mode = 'val'

    # data loader
    sys.path.append(f'../{modality}/')
    from data.data_loader import get_dataloader
    if modality == "ct":
        noise_level = kwargs.get('noise_level', 2.0)
        data_loader = get_dataloader(mode, noise_level=noise_level)
    elif modality == "mri":
        coil_type = kwargs.get('coil_type', 'single')
        acc = kwargs.get('acc', 4)
        cf = kwargs.get('cf', 0.08)
        noise_sd = kwargs.get('noise_sd', 0.0)
        data_type = kwargs.get('data_type', 'pd')
        data_loader = get_dataloader(mode, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type)
    else:
        raise ValueError(f"modality: {modality} not recognized")

    
    # reconstruction map
    reconstruction_map = get_reconstruction_map(method, modality, device=device, **kwargs)

    # reconstruction map wrapper for batch
    n_hyperparameters = kwargs.get('n_hyperparameters', 2)
    reconstruction_map_wrapper = ReconstructionMap(reconstruction_map, data_loader, n_hyperparameters=n_hyperparameters, device=device, modality=modality)

    # get directory name of current file
    dir_name = os.path.dirname(__file__)
    dir_name = f'{os.path.dirname(__file__)}/../{modality}/validation_data'


    # if folder does not exist, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    freeze_p2 = (reconstruction_map_wrapper.n_hyperparameters == 1)

    # extra parameters
    gamma_stop = kwargs.get('gamma_stop', 1.05)

    kwargs['gamma_stop'] = gamma_stop


    validator = ValidateCoarseToFine(reconstruction_map_wrapper.batch_score, dir_name=dir_name, exp_name=job_name, freeze_p2=freeze_p2, **kwargs)
    validator.run()
    

def test(method, modality, device, job_name, **kwargs):

    mode = 'test'

    # get directory name of current file
    dir_name = f'{os.path.dirname(__file__)}/../{modality}/test_data'
    dir_name_val = f'{os.path.dirname(__file__)}/../{modality}/validation_data'

    # if folder does not exist, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_name_recon = f'{os.path.dirname(__file__)}/../{modality}/test_data/recon/{job_name}'
    if not os.path.exists(dir_name_recon):
        os.makedirs(dir_name_recon)


    # val data to recover hyperparameters
    try:
       val_data = pd.read_csv(f"{dir_name_val}/validation_scores_{job_name}.csv").reset_index(drop=True)
    except:
        raise ValueError(f"validation_file_name: validation_scores_{job_name}.csv not found")
    
    # get hyperparameters
    p1 = val_data.loc[val_data["psnr"].idxmax()]["p1"]
    p2 = val_data.loc[val_data["psnr"].idxmax()]["p2"]

    print(f"Optimal validation hyperparameters p1: {p1}, p2: {p2}")

    # data loader
    sys.path.append(f'../{modality}/')
    from data.data_loader import get_dataloader
    if modality == "ct":
        noise_level = kwargs.get('noise_level', 2.0)
        data_loader = get_dataloader(mode, noise_level=noise_level)
    elif modality == "mri":
        coil_type = kwargs.get('coil_type', 'single')
        acc = kwargs.get('acc', 4)
        cf = kwargs.get('cf', 0.08)
        noise_sd = kwargs.get('noise_sd', 0.0)
        data_type = kwargs.get('data_type', 'mixed')
        data_loader = get_dataloader(mode, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type)
    else:
        raise ValueError(f"modality: {modality} not recognized")

    # reconstruction map
    reconstruction_map = get_reconstruction_map(method, modality, device=device, **kwargs)

   

    # reconstruction map wrapper for batch
    n_hyperparameters = kwargs.get('n_hyperparameters', 2)

    reconstruction_map_wrapper = ReconstructionMap(reconstruction_map, data_loader, n_hyperparameters=n_hyperparameters, device=device, modality=modality, export=True, num_exports=10, export_name=f'{dir_name_recon}/{job_name}', mode = 'test')


    psnr_, ssim_, n_iter_ = reconstruction_map_wrapper.batch_score(p1, p2)

    cols = ["p1", "p2", "job_name", "psnr", "ssim", "niter"]

    data = [[p1, p2, job_name, psnr_, ssim_, n_iter_]]
    df = pd.DataFrame(data, columns=cols)

    df.to_csv(f"{dir_name}/{job_name}.csv")


  


class ReconstructionMap():
    def __init__(self, sample_reconstruction_map, data_loader, n_hyperparameters=2, modality="ct", device='cuda:0', **kwargs):

        self.sample_reconstruction_map = sample_reconstruction_map
        self.data_loader = data_loader
        self.n_hyperparameters = n_hyperparameters
        self.device = device
        self.modality = modality

        self.mode = kwargs.get("mode", "val")
        self.export = kwargs.get("export", False)
        self.num_exports = kwargs.get("num_exports", 10)
        self.export_name = kwargs.get("export_name", "test")

        if n_hyperparameters > 2 or n_hyperparameters < 1:
            raise ValueError("n_hyperparameters must be 1 or 2")
        
    def batch_score(self, p1, p2=None):

        data_loader = self.data_loader
        psnr_val = torch.zeros(len(data_loader))
        ssim_val = torch.zeros(len(data_loader))
        n_iter_val = torch.zeros(len(data_loader))

        for idx, batch in enumerate(data_loader):
            if self.modality == "ct":
                phantom = batch["phantom"].to(self.device)
                sinogram = batch["sinogram"].to(self.device)
                fbp = batch["fbp"].to(self.device)
            
                if self.n_hyperparameters == 1:
                    x, psnr_, ssim_, n_iter_ = self.sample_reconstruction_map(sinogram, p1, x_gt=phantom, x_init=fbp)
                else:
                    x, psnr_, ssim_, n_iter_ = self.sample_reconstruction_map(sinogram, p1, p2, x_gt=phantom, x_init=fbp)

                if self.export:
                    if idx % (len(data_loader)//self.num_exports) == 0:
                        x_np = x.squeeze().detach().cpu().numpy()
                        np.save(f"{self.export_name}_x_{idx}.npy", x_np)

            elif self.modality == "mri":
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)
                if "smaps" in batch:
                    smaps = batch["smaps"].to(self.device)
                else:
                    smaps = None
            
                if self.n_hyperparameters == 1:
                    x, psnr_, ssim_, n_iter_ = self.sample_reconstruction_map(y, mask, smaps, p1, x_gt=x)
                else:
                    x, psnr_, ssim_, n_iter_ = self.sample_reconstruction_map(y, mask, smaps, p1, p2, x_gt=x)

                if self.export:
                    if idx % (len(data_loader)//self.num_exports) == 0:
                        x_np = x.squeeze().detach().cpu().numpy()
                        np.save(f"{self.export_name}_x_{idx}.npy", x_np)


            psnr_val[idx] = psnr_
            n_iter_val[idx] = n_iter_
            ssim_val[idx] = ssim_
            if self.mode == "test":
                 print(f" {idx+1}/{len(data_loader)} ==> average: psnr {psnr_val[:idx+1].mean().item():.2f}, ssim {ssim_val[:idx+1].mean().item():.3f})")
        return(psnr_val.mean().item(), ssim_val.mean().item(), n_iter_val.mean().item())

        

