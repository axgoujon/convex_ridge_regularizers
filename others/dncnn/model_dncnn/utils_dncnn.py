
import torch
import torch.nn as nn
import numpy as np
import os
from realSN_models import DnCNN

# ---- load the model based on the type and sigma (noise level) ---- 
def load_model(sigma, device):
    path = os.path.dirname(__file__)

    path = f"{path}/../pretrained_models/RealSN_DnCNN" + "_noise" + str(sigma) + ".pth"
    
    net = DnCNN(channels=1, num_of_layers=17)

    model = nn.DataParallel(net, device_ids=[int(device.split(":")[-1])]).to(device)
    
    model.load_state_dict(torch.load(path))
    model.eval()
    return model