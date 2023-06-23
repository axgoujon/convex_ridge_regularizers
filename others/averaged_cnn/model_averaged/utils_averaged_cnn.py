import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(__file__))

from averaged_cnn import AveragedCNN
from lipschitzconv2d import LipschitzConv2d

def load_model(sigma=5, device="cuda:0"):
    path = os.path.dirname(__file__)

    path = f"{path}/../pretrained_models/sigma_{sigma}/checkpoints/checkpoint_best_epoch.pth"


    cnn_model = AveragedCNN()

    cnn_model.to(device)
    for module in cnn_model.layers:
            if isinstance(module, LipschitzConv2d):
                    module.additional_parameters['largest_eigenvector'] = module.additional_parameters['largest_eigenvector'].to(device)
            
    print("Number of parameters in the model: ", cnn_model.num_params)

    checkpoint = torch.load(path, device)
    cnn_model.load_state_dict(checkpoint['state_dict'], strict=True)

    cnn_model = cnn_model.eval()

    cnn_model.to(device)

    return(cnn_model)
