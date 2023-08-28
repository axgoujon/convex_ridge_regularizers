import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'ct'
method = 'wcrr'
n_hyperparameters = 2

# constrain the grid search
p2_max = 30
p2_init = 3
p1_init = 3000

p1_init = 1409
p2_init = 1.51
tol = 5e-6

if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [0.5, 1.0, 2.0]

    NOISE_LEVELS = [0.5, 1.0, 2.0]

    EXP_NAMES = ['WCRR-CNN']


    for exp_name in EXP_NAMES:
        for noise_level in NOISE_LEVELS:
            job_name = f"{modality}_noise_{noise_level}_wcrr_{exp_name}_"
            validate(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, model_name = exp_name, n_hyperparameters=n_hyperparameters,\
                     p2_max=p2_max, p2_init=p2_init, p1_init=p1_init, tol=tol)

