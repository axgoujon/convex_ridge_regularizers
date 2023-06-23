import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'ct'
method = 'pnp'
n_hyperparameters = 2

# constrain the grid search
# relaxation parameter
p1_max = 1
p1_init = 0.03

# step size parameter
p2_max = 2
p2_init = 1

tol = 1e-5
max_iter = 2000


if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    NOISE_LEVELS = [1.0, 2.0]

    MODEL = [['dncnn', 5],['averaged_cnn', 5],  ['dncnn', 15], ['averaged_cnn', 25]]

    opt = 101

    if opt == 0:
        MODEL = [['averaged_cnn', 5], ['averaged_cnn', 25]]
        NOISE_LEVELS = [1.0]
    elif opt == 1:
        MODEL = [['averaged_cnn', 5], ['averaged_cnn', 25]]
        NOISE_LEVELS = [2.0]
    elif opt == 2:
        MODEL = [['dncnn', 5], ['dncnn', 15]]
        NOISE_LEVELS = [1.0, 2.0]
    elif opt == 100:
        MODEL = [['averaged_cnn', 5]]
        NOISE_LEVELS = [1.0]
    elif opt == 101:
        MODEL = [['averaged_cnn', 5]]
        NOISE_LEVELS = [2.0]
    else:
        raise ValueError(f"opt {opt} not supported")

    MODEL = [['dncnn', 5]]
    NOISE_LEVELS = [0.5]
    
    for model in MODEL:
        for noise_level in NOISE_LEVELS:
            job_name = f"{modality}_noise_{noise_level}_pnp_{model[0]}_sig_train_{model[1]}"
            validate(method = method, modality = modality, job_name=job_name, noise_level=noise_level, device=device, model_type = model[0], model_sigma=model[1], n_hyperparameters=n_hyperparameters,\
                     p1_max=p1_max, p2_max=p2_max, p1_init=p1_init, p2_init=p2_init, tol=tol, max_iter=max_iter)

