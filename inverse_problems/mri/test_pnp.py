import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import test

modality = 'mri'
method = 'pnp'
n_hyperparameters = 2

tol = 5e-7
max_iter = 10000

if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    model = ['averaged_cnn', 5]
    opt = '2'
    data_type = 'pdfs'
    noise_sd = 0.002

    if (opt == '1'):
        coil_type = 'single'
        acc = 2
        cf = 0.16

    elif (opt == '2'):
        coil_type = 'single'
        acc = 4
        cf = 0.08

    elif (opt == '3'):
        coil_type = 'multi'
        acc = 4
        cf = 0.08

    elif (opt == '4'):
        coil_type = 'multi'
        acc = 8
        cf = 0.04
    

    job_name = f"{modality}_coiltype_{coil_type}_acc_{acc}_cf_{cf}_noisesd_{noise_sd}_datatype_{data_type}_pnp_{model[0]}_sig_train_{model[1]}"
    test(method = method, modality = modality, job_name=job_name, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type, device=device, model_type = model[0], model_sigma=model[1], n_hyperparameters=n_hyperparameters, tol=tol, max_iter=max_iter, crop=True)


