import argparse
import time
import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import test

modality = 'ct'
method = 'acr'
n_hyperparameters = 2


if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [0.5]
    time.sleep(30000)
    for noise_level in NOISE_LEVELS:
        job_name = f"{modality}_noise_{noise_level}_acr_True"
        test(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, n_hyperparameters=n_hyperparameters)

