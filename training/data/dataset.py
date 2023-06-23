import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
import glob
import os
from torchvision import transforms
from PIL import Image

random.seed(42)

class H5PY(Dataset):
    def __init__(self, data_file, randomize=True):
        super(Dataset, self).__init__()
        self.data_file = data_file
        self.dataset = None
        with h5py.File(self.data_file, 'r') as file:
            self.keys_list = list(file.keys())
            if randomize:
                random.shuffle(self.keys_list)


    def __len__(self):
        return len(self.keys_list)


    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.data_file, 'r')
        data = torch.Tensor(np.array(self.dataset[self.keys_list[idx]]))
        return data
    



