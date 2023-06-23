import os
import numpy as np
import h5py
import shutil
from fastmri.data import subsample
import torch
import torch.nn as nn
import sys
sys.path.insert(0, './bart-0.8.00/python')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = './bart-0.8.00'
sys.path.append('./bart-0.8.00/python')
from bart import bart

torch.manual_seed(0)
np.random.seed(0)


def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


device = 'cpu'

center_fraction = 0.04
acceleration = 8
noise_std_dev = 0.002

gen_sc = False
gen_mc = True

mask_func = subsample.RandomMaskFunc(center_fractions=[center_fraction], accelerations=[acceleration])

loaddata_path = '/home/bohra/MRI_exps/data/multicoil_val_clean'

if (gen_sc):
    savedata_path_sc = '/home/bohra/MRI_exps/simdata/singlecoil_acc_' + str(acceleration) + '_cf_' + str(center_fraction) + '_noisesd_' + str(noise_std_dev)
    shutil.rmtree(savedata_path_sc, ignore_errors=True)
    os.makedirs(savedata_path_sc, exist_ok=True)

if (gen_mc):
    savedata_path_mc = '/home/bohra/MRI_exps/simdata/multicoil_acc_' + str(acceleration) + '_cf_' + str(center_fraction) + '_noisesd_' + str(noise_std_dev)
    shutil.rmtree(savedata_path_mc, ignore_errors=True)
    os.makedirs(savedata_path_mc, exist_ok=True)

files = sorted(os.listdir(loaddata_path))
total_num_files = len(files)
num_val_files_each = 10
num_test_files_each = 50


print('Generating datasets')
count_pd_val = 0
count_pd_test = 0
count_pdfs_val = 0
count_pdfs_test = 0

for file_idx in range(total_num_files):
    print('File idx: ', file_idx)
    loadfilename = os.path.join(loaddata_path, files[file_idx])
    filecontent = h5py.File(loadfilename, 'r')

    kspace = filecontent['kspace']
    slice_idx = kspace.shape[0]//2
    kspace = np.expand_dims(kspace[slice_idx,:,:,:], axis=0)
    kspace_bart = np.transpose(kspace, (0, 2, 3, 1))
    kspace = torch.tensor(kspace)

    # Use fully-sampled kspace data to create ground-truth images
    kspace_shifted = torch.fft.ifftshift(kspace, dim=(2,3))
    ifft_kspace_shifted = torch.fft.ifft2(kspace_shifted, dim=(2,3), norm='ortho')
    ifft_kspace = torch.fft.fftshift(ifft_kspace_shifted, dim=(2,3))
    x = torch.sqrt(torch.sum(torch.abs(ifft_kspace)**2, dim=1, keepdim=True))

    # Normalize x
    x = x/torch.max(x)

    # Cropped version of
    x_crop = center_crop(x, [320,320])

    # Create Mask
    h = kspace.shape[2]
    w = kspace.shape[3]
    mask, num_low_frequencies = mask_func([h, w, 1])
    mask = mask[:,:,0]
    M = mask.to(device)
    M = M.expand(h,-1)
    M = M.unsqueeze(0)
    M = M.unsqueeze(0)
    print('Shape of mask: ', M.shape)
    M = torch.fft.ifftshift(M, dim=(2,3))

    if (dict(filecontent.attrs)['acquisition'] == 'CORPD_FBK'):
        data_tag = '/pd'
        if (count_pd_val < num_val_files_each):
            save_tag = '/val_images/'
            count_pd_val = count_pd_val + 1
        elif (count_pd_test < num_test_files_each):
            save_tag = '/test_images/'
            count_pd_test = count_pd_test + 1
        else:
            save_tag = '/train_images/'

    elif (dict(filecontent.attrs)['acquisition'] == 'CORPDFS_FBK'):
        data_tag = '/pdfs'
        if (count_pdfs_val < num_val_files_each):
            save_tag = '/val_images/'
            count_pdfs_val = count_pdfs_val + 1
        elif (count_pdfs_test < num_test_files_each):
            save_tag = '/test_images/'
            count_pdfs_test = count_pdfs_test + 1
        else:
            save_tag = '/train_images/'


    if (gen_sc):
        # Generate singlecoil measurements
        y0_sc = torch.fft.fft2(x, dim=(2,3), norm='ortho')*M
        y_sc = y0_sc + (noise_std_dev*torch.randn(y0_sc.shape, device=device) + 1j*noise_std_dev*torch.randn(y0_sc.shape, device=device))
        y_sc = y_sc*M  # mask noise

        snr_meas_sc = 10*torch.log10(torch.mean(torch.abs(y0_sc.cpu())**2)/torch.mean((torch.abs(y_sc.cpu() - y0_sc.cpu()))**2))
        print('Measurements: SNR_SC = {:.6f}\t'.format(snr_meas_sc))

        currfile_savepath_sc = savedata_path_sc + data_tag + save_tag + files[file_idx][0:11]
        os.makedirs(currfile_savepath_sc, exist_ok=True)
        torch.save(x, currfile_savepath_sc + '/x.pt')
        torch.save(x_crop, currfile_savepath_sc + '/x_crop.pt')
        torch.save(y_sc, currfile_savepath_sc + '/y.pt')
        torch.save(M, currfile_savepath_sc + '/mask.pt')


    if (gen_mc):
        smaps = bart(1, "ecalib -m1 -W -c0", kspace_bart)
        smaps = np.transpose(smaps, (0,3,1,2))
        smaps = torch.tensor(smaps)
        # Generate multicoil measurements
        y0_mc = torch.fft.fft2(x*smaps, dim=(2,3), norm='ortho')*M
        y_mc = y0_mc + (noise_std_dev*torch.randn(y0_mc.shape, device=device) + 1j*noise_std_dev*torch.randn(y0_mc.shape, device=device))
        y_mc = y_mc*M  # mask noise

        snr_meas_mc = 10*torch.log10(torch.mean(torch.abs(y0_mc.cpu())**2)/torch.mean((torch.abs(y_mc.cpu() - y0_mc.cpu()))**2))
        print('Measurements: SNR_MC = {:.6f}\t'.format(snr_meas_mc))

        currfile_savepath_mc = savedata_path_mc + data_tag + save_tag + files[file_idx][0:11]
        os.makedirs(currfile_savepath_mc, exist_ok=True)
        torch.save(x, currfile_savepath_mc + '/x.pt')
        torch.save(x_crop, currfile_savepath_mc + '/x_crop.pt')
        torch.save(y_mc, currfile_savepath_mc + '/y.pt')
        torch.save(M, currfile_savepath_mc + '/mask.pt')
        torch.save(smaps, currfile_savepath_mc + '/smaps.pt')