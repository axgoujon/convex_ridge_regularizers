import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'mri'
method = 'tv'
n_hyperparameters = 1

tol = 1e-5
max_iter = 3000
# constrain the grid search
# relaxation parameter
#p1_init = 0.0005
#p1_init = 2*1e-5
p1_init = 0.0006194895
gamma_stop = 1.1




if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    opt = '1'
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
    
    job_name = f"{modality}_coiltype_{coil_type}_acc_{acc}_cf_{cf}_noisesd_{noise_sd}_datatype_{data_type}_tv_debug"
    validate(method = method, modality = modality, job_name=job_name, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type, device=device, n_hyperparameters=n_hyperparameters,\
    p1_init=p1_init, tol=tol, max_iter=max_iter, gamma_stop=gamma_stop, crop=True)
