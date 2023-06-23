import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'ct'
method = 'tv'
n_hyperparameters = 1

tol = 5e-5
max_iter = 2000
# constrain the grid search
# relaxation parameter
p1_init = 30

gamma_stop = 1.01




if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [0.5]

    for noise_level in NOISE_LEVELS:
        job_name = f"{modality}_noise_{noise_level}_tv"
        validate(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, n_hyperparameters=n_hyperparameters,\
                    p1_init=p1_init, tol=tol, max_iter=max_iter, gamma_stop=gamma_stop)

