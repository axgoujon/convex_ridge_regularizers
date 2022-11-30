import json
import os
import glob
import sys
import torch
sys.path.append('../')
from models.convex_ridge_regularizer import ConvexRidgeRegularizer
from pathlib import Path

def load_model(name, device='cuda:0', epoch=None):
    # folder
    directory = f'./trained_models/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    # retrieve last checkpoint if epich not specified
    if epoch is None:
        files = glob.glob(f'{directory}checkpoints/*.pth', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]), files)
        epoch = max(epochs)
        print(f"--- loading checkpoint from epoch {epoch} ---")

    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.pth'
    # config file
    config = json.load(open(f'{directory}config.json'))
    # build model
    model, _ = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return(model)

def build_model(config):
    model = ConvexRidgeRegularizer(kernel_size=config['net_params']['kernel_size'],
                            channels = config['net_params']['channels'],
                            padding=config['net_params']['padding'],
                            activation_params=config['activation_params'])

    return(model, config)