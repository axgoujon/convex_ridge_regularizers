import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import test

modality = 'ct'
method = 'pnp'
n_hyperparameters = 2

tol = 5e-6
max_iter = 2000

if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [1.0, 2.0]

    MODEL = [['dncnn', 5], ['averaged_cnn', 5], ['dncnn', 15], ['averaged_cnn', 25]]

    NOISE_LEVELS = [0.5]
    MODEL = [['dncnn', 5]]

    #MODEL = [['averaged_cnn', 5]]
    
    for model in MODEL:
        for noise_level in NOISE_LEVELS:
            job_name = f"{modality}_noise_{noise_level}_pnp_{model[0]}_sig_train_{model[1]}"
            test(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, model_type = model[0], model_sigma=model[1], n_hyperparameters=n_hyperparameters, tol=tol, max_iter=max_iter)

