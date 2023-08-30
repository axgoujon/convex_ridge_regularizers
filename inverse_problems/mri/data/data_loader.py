import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
import random
from PIL import Image

class fastmri_dataset_multi(Dataset):
    def __init__(self, mode = 'val', acc = 4, cf = 0.08, noise_sd = 0.0, data_type = 'pd'):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(dir_path, 'data_sets/multicoil_acc_' + str(acc) + '_cf_' + str(cf) + '_noisesd_' + str(noise_sd) , data_type, mode + '_images')
        self.data_files = sorted(os.listdir(self.data_dir))


    def __getitem__(self, index):
        curr_file_dir = os.path.join(self.data_dir, self.data_files[index])
        mask = torch.load(os.path.join(curr_file_dir, 'mask.pt'))[0,:,:,:]
        smaps = torch.load(os.path.join(curr_file_dir, 'smaps.pt'))[0,:,:,:]
        y = torch.load(os.path.join(curr_file_dir, 'y.pt'))[0,:,:,:]
        x = torch.load(os.path.join(curr_file_dir, 'x_crop.pt'))[0,:,:,:]

        return {'mask': mask, 'smaps': smaps, 'y': y, 'x': x}


    def __len__(self):
        return len(self.data_files)


class fastmri_dataset_single(Dataset):
    def __init__(self, mode = 'val', acc = 4, cf = 0.08, noise_sd = 0.0, data_type = 'pd'):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.data_dir = os.path.join(dir_path, 'data_sets/singlecoil_acc_' + str(acc) + '_cf_' + str(cf) + '_noisesd_' + str(noise_sd) , data_type, mode + '_images/')

    
        self.data_files = sorted(os.listdir(self.data_dir))

        #

    def __getitem__(self, index):
        curr_file_dir = os.path.join(self.data_dir, self.data_files[index])
        mask = torch.load(os.path.join(curr_file_dir, 'mask.pt'))[0,:,:,:]
        y = torch.load(os.path.join(curr_file_dir, 'y.pt'))[0,:,:,:]
        x = torch.load(os.path.join(curr_file_dir, 'x_crop.pt'))[0,:,:,:]

        return {'mask': mask, 'y': y, 'x': x}


    def __len__(self):
        return len(self.data_files)
    


def get_dataloader(mode, coil_type = 'single', acc = 0.4, cf = 0.08, noise_sd = 0.0, data_type = 'pd'):
    if (coil_type == 'single'):
        dataloader = DataLoader(fastmri_dataset_single(mode=mode, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type),\
                                batch_size = 1, shuffle = False)
    elif (coil_type == 'multi'):
        dataloader = DataLoader(fastmri_dataset_multi(mode=mode, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type),\
                                batch_size = 1, shuffle = False)
    return dataloader
