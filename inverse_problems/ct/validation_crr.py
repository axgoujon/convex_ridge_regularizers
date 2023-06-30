import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'ct'
method = 'crr'
n_hyperparameters = 2

# constrain the grid search
p2_max = 50
p2_init = 2
p1_init = 0.05

tol = 5e-6

if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    # measurement noise levels, corresponding data set must be made before 
    NOISE_LEVELS = [0.5, 1.0, 2.0]

    # CRR-NN models to validate

    # full list ['CT_Sigma_5_t_10', 'CT_Sigma_25_t_10', 'Sigma_25_t_1',  'Sigma_25_t_10', 'Sigma_25_t_50', 'Sigma_5_t_1', 'Sigma_5_t_10', 'Sigma_5_t_50']
    EXP_NAMES = ['CT_Sigma_5_t_10']

    for exp_name in EXP_NAMES:
        for noise_level in NOISE_LEVELS:
            job_name = f"{modality}_noise_{noise_level}_crr_{exp_name}"
            validate(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, model_name = exp_name, n_hyperparameters=n_hyperparameters,\
                     p2_max=p2_max, p2_init=p2_init, p1_init=p1_init, tol=tol)

