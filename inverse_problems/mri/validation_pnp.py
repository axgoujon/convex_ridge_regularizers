import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'mri'
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
gamma_stop = 1.1


if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device

    #MODELS = [['dncnn', 5],['averaged_cnn', 5],  ['dncnn', 15], ['averaged_cnn', 25]]
    #model = ['averaged_cnn', 5]
    #model = ['averaged_cnn', 25]
    #model = ['dncnn', 5]
    model = ['dncnn', 15]
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
    

    job_name = f"{modality}_coiltype_{coil_type}_acc_{acc}_cf_{cf}_noisesd_{noise_sd}_datatype_{data_type}_pnp_{model[0]}_sig_train_{model[1]}"
    validate(method = method, modality = modality, job_name=job_name, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type, device=device, model_type = model[0], model_sigma=model[1], n_hyperparameters=n_hyperparameters,\
    p1_max=p1_max, p2_max=p2_max, p1_init=p1_init, p2_init=p2_init, tol=tol, max_iter=max_iter, crop=True, gamma_stop=gamma_stop)


