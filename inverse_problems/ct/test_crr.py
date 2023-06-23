import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import test

modality = 'ct'
method = 'crr'
n_hyperparameters = 2
algo_name = 'adagd'

if algo_name == 'fista':
    tol = 1e-5
else:
    tol = 1e-6

if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [1.0, 2.0]

    opt = 1

    if opt == 0:
        NOISE_LEVELS = [1.0]
    elif opt == 1:
        NOISE_LEVELS = [2.0]
    else:
        raise ValueError("opt must be 0 or 1")

    EXP_NAMES = ['CT_Sigma_5_t_10', 'CT_Sigma_25_t_10', 'Sigma_25_t_1',  'Sigma_25_t_10', 'Sigma_25_t_50', 'Sigma_5_t_1', 'Sigma_5_t_10', 'Sigma_5_t_50']

    # EXP_NAMES = ['Sigma_5_t_50']
    # NOISE_LEVELS = [2.0]

    NOISE_LEVELS = [0.5]
    EXP_NAMES = EXP_NAMES[0:4]
    EXP_NAMES = ['CT_Sigma_25_t_10']
    #EXP_NAMES = EXP_NAMES[4:]
    
    for exp_name in EXP_NAMES:
        for noise_level in NOISE_LEVELS:
            job_name = f"{modality}_noise_{noise_level}_crr_{exp_name}"
            test(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, model_name = exp_name, n_hyperparameters=n_hyperparameters, tol=tol, algo_name=algo_name)

