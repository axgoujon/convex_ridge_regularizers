import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
import random
from PIL import Image

class mayo_dataset(Dataset):
    def __init__(self, mode = 'train', noise_level = 2.0):
        root = os.path.dirname(__file__) + "/data_sets"
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.files_sinogram = sorted(glob.glob(os.path.join(root, f'{mode}/Sinogram/*_sig_{noise_level:.1f}*.npy')))
        self.files_fbp = sorted(glob.glob(os.path.join(root, f'{mode}/FBP/*_sig_{noise_level:.1f}*.npy')))
        self.files_phantom = sorted(glob.glob(os.path.join(root, f'{mode}/Phantom/*.npy')))


    def __getitem__(self, index):
        sinogram = self.transform(Image.fromarray(np.load(self.files_sinogram[index % len(self.files_sinogram)])))
        fbp = self.transform(Image.fromarray(np.load(self.files_fbp[index % len(self.files_fbp)])))
        
        phantom = self.transform(Image.fromarray(np.load(self.files_phantom[index % len(self.files_phantom)])))

        return {'fbp': fbp, 'phantom': phantom, 'sinogram': sinogram}

    def __len__(self):
        return max([len(self.files_phantom), len(self.files_fbp), len(self.files_sinogram)])
    

def get_dataloader(mode, noise_level = 2.0):
    dataloader = DataLoader(mayo_dataset(mode = mode, noise_level=noise_level),\
                              batch_size = 1, shuffle = False)
    
    return dataloader
