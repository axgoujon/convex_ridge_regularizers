import os
import numpy as np
import torch
import torch.nn as nn
import sys
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


device = 'cuda:0'
coil_type = 'multicoil'
acc = 8
cf = 0.04
noise_sd = 0.002
data_type = 'pdfs'
mode = 'test'

data_folder = '/home/bohra/MRI_exps/simdata/' + coil_type + '_acc_' + str(acc) + '_cf_' + str(cf) + '_noisesd_' + str(noise_sd) + '/' + data_type + '/' + mode + '_images'
print(data_folder)
files = sorted(os.listdir(data_folder))
num_files = len(files)

if (coil_type == 'singlecoil'):
    psnr_sum = 0.0
    ssim_sum = 0.0
    for file_idx in range(num_files):
        filepath = os.path.join(data_folder, files[file_idx])
        x = torch.load(filepath + '/x_crop.pt')
        y = torch.load(filepath + '/y.pt')
        mask = torch.load(filepath + '/mask.pt')
        x = x.to(device)
        y = y.to(device)
        mask = mask.long().to(device)

        hty = center_crop(torch.real(torch.fft.ifft2(y*mask, norm='ortho')), [320,320])
        psnr_sum = psnr_sum + (psnr(hty, x, data_range=1.0)).item()
        ssim_sum = ssim_sum + (ssim(hty, x, data_range=1.0)).item()

    avg_psnr = psnr_sum/num_files
    avg_ssim = ssim_sum/num_files

    print('Avg psnr: ', avg_psnr)
    print('Avg ssim: ', avg_ssim)


elif (coil_type == 'multicoil'):
    hty_psnr_sum = 0.0
    hty_ssim_sum = 0.0
    rss_psnr_sum = 0.0
    rss_ssim_sum = 0.0
    for file_idx in range(num_files):
        filepath = os.path.join(data_folder, files[file_idx])
        x = torch.load(filepath + '/x_crop.pt')
        y = torch.load(filepath + '/y.pt')
        mask = torch.load(filepath + '/mask.pt')
        smaps = torch.load(filepath + '/smaps.pt')
        x = x.to(device)
        y = y.to(device)
        mask = mask.long().to(device)
        smaps = smaps.to(device)

        hty = center_crop(torch.sum(torch.real(torch.fft.ifft2(y*mask, norm='ortho')*torch.conj(smaps)), dim=1, keepdim=True), [320,320])
        hty_psnr_sum = hty_psnr_sum + (psnr(hty, x, data_range=1.0)).item()
        hty_ssim_sum = hty_ssim_sum + (ssim(hty, x, data_range=1.0)).item()

        zf_ifft = torch.fft.ifft2(y, dim=(2,3), norm='ortho')
        x_rss = center_crop(torch.sqrt(torch.sum(torch.abs(zf_ifft)**2, dim=1, keepdim=True)), [320,320])
        rss_psnr_sum = rss_psnr_sum + (psnr(x_rss, x, data_range=1.0)).item()
        rss_ssim_sum = rss_ssim_sum + (ssim(x_rss, x, data_range=1.0)).item()

    avg_hty_psnr = hty_psnr_sum/num_files
    avg_hty_ssim = hty_ssim_sum/num_files

    print('Avg HTy psnr: ', avg_hty_psnr)
    print('Avg HTy ssim: ', avg_hty_ssim)

    avg_rss_psnr = rss_psnr_sum/num_files
    avg_rss_ssim = rss_ssim_sum/num_files

    print('Avg RSS psnr: ', avg_rss_psnr)
    print('Avg RSS ssim: ', avg_rss_ssim)




