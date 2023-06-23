import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'ct'
method = 'acr'
n_hyperparameters = 2

# constrain the grid search
# relaxation parameter
p1_init = 2000
p2_init = 1e-5



if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [0.5]


    for noise_level in NOISE_LEVELS:
        job_name = f"{modality}_noise_{noise_level}_acr_True"
        validate(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, n_hyperparameters=n_hyperparameters,\
                    p1_init=p1_init, p2_init=p2_init)

