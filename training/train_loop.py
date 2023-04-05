import argparse
import json
import torch
import trainer
import copy
import os
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def main(device):
    """ Training multiple models with different t-steps"""
    config_path = 'configs/config.json'
    config_init = json.load(open(config_path))
    
    # loop over t-step training
    for t in [1, 2, 5, 10, 20, 30, 50]:
        print(f"**** {t}-step(s) training *****","\n")
        config = copy.deepcopy(config_init)
        config["training_options"]["t_steps"] = t
        config['exp_name'] = f"{config['exp_name']}_t_{t}"

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
