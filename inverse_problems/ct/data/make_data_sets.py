import numpy as np
import torch

import sys
sys.path += ['..', '../utils/']
from utils_ct.ct_forward_utils import get_operators

import os

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

device = "cuda"


skip_train = True # compared to original code we don't train on the ct images
torch.manual_seed(0)


# get the operators
fwd_op, fbp_op, adjoint_op = get_operators()

noise_levels = [0.5, 1.0, 2.0]
####################arrange the slices into training and test data ###############################
if __name__ == '__main__':
    print('creating training, validation and test set')
    datapath = './mayo_data/'
    output_datapath = './data_sets/'
    
    files = sorted(os.listdir(datapath))
    
    for sig in noise_levels:
        psnr_fbp_test = []
        ssim_fbp_test = []
        for idx in range(len(files)):
            filename = datapath + files[idx]
            
            #####use patient L109 for testing, rest for training and validation
            if('L109' not in filename): 
                if idx % 50 == 0 and idx < 300:
                    mode = 'val'
                else:
                    mode = 'train'
            else:
                mode = 'test'

            if not skip_train or (mode != 'train'):

                image = np.load(filename, allow_pickle=True)
                image = (image - image.min())/(image.max() - image.min()) #normalize range to [0,1]

                ###compute projection and FBP #############
                phantom = torch.from_numpy(image).view(1, 1, image.shape[0], image.shape[1]).to(device)
                sinogram = fwd_op(phantom)

                noise = sig * torch.randn_like(sinogram)

                sinogram_noisy = sinogram + noise

                fbp = fbp_op(sinogram_noisy)


                ######save the images as numpy files###############
                sinogram_image = sinogram_noisy.cpu().numpy().squeeze()
                fbp_image = fbp.cpu().numpy().squeeze()
                psnr = compare_psnr(image, fbp_image, data_range=1.0)
                ssim = compare_ssim(image, fbp_image, data_range=1.0)
                nmse = torch.mean((phantom - fbp)**2)/torch.mean(phantom**2)

                print('FBP: NSME = {:.6f}\t PSNR = {:.6f}'.format(nmse, psnr))

                psnr_fbp_test.append(psnr)
                ssim_fbp_test.append(ssim)

                #####save phantom#####
                path = f'{output_datapath}{mode}/Phantom/'
                out_filename = path + 'phantom_%d'%idx + '.npy'
                os.makedirs(path, exist_ok=True)
                np.save(out_filename, image)
                
                #####save FBP#####
                path = f'{output_datapath}{mode}/FBP/'
                out_filename = f'{path}fbp_sig_{sig:.1f}_{idx}.npy'
                os.makedirs(path, exist_ok=True)
                np.save(out_filename, fbp_image)
                
                #####save sinogram#####
                path = f'{output_datapath}{mode}/Sinogram/'
                out_filename = f'{path}sinogram_sig_{sig:.1f}_{idx}.npy'
                os.makedirs(path, exist_ok=True)
                np.save(out_filename, sinogram_image)
            
        print('FBP: PSNR = {:.6f}'.format(np.mean(psnr_fbp_test)))
        print('FBP: SSIM = {:.6f}'.format(np.mean(ssim_fbp_test)))
            
