import argparse
import json
import torch
import trainer
import os

# CUDA_LAUNCH_BLOCKING=1
def main(device):
    
    # Set up directories for saving results

    config_path = 'configs/config.json'
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    trainer_inst = trainer.Trainer(config, seed, device)
   
    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cuda", type=str,
                        help='device to use')
    args = parser.parse_args()

    main(args.device)
