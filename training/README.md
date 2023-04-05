## Data Preparation
(mostly taken from [link](https://github.com/uclaopt/Provable_Plug_and_Play/tree/master/training)).

The BSD images are provided in the data folder.

To generate the preprocessed ```.h5``` data set files (data augmentation + patch extraction for training):

```console
~/convex_ridge_regularizers/training/data$ python BSD_preprocessing.py
```

## Training
To launch the training on, e.g., GPU \#0:
```console
~/convex_ridge_regularizers/training$ python train.py --device cuda:0
```

Follow the training with tensorboard:
```console
~/convex_ridge_regularizers/trained_models$ tensorboard --logdir .
```
The training parameters can be accessed/modified in ```/configs/config.json```.
 Default parameters:
```json
{   "exp_name": "Test_Sigma_5_t1", //name of the experiment for saving the model
    "sigma": 5, // noise level for training
    "activation_params": {
        "knots_range": 0.1, //the splines breakpoints are between -x and x
        "n_knots": 21, //number of knots
    },
    "net_params": {
        "channels": [
            1,
            8,
            32
        ],// channels for the multiconvolution layer
        "kernel_size": 7
    },
    "optimizer": {
        "lr_conv": 0.001, // learning rate for the convolutional layer
        "lr_lmbd": 0.05, // " " " for the reg param lambda
        "lr_mu": 0.05, // " " " for the scaling param mu
        "lr_spline_coefficients": 5e-05 // " " " typically small
    },
    "train_dataloader": {
        "batch_size": 128,
        "num_workers": 1,
        "shuffle": true,
        "train_data_file": "data/preprocessed/train_BSD_40.h5"
    },
    "training_options": {
        "epochs": 10,
        "t_steps": 1, // training a x-step denoiser
        "lr_scheduler": {
            "gamma": 0.75,
            "nb_steps": 10,
            "use": true
        },
        "tv2_lmbda": 2e-3 // spline regularization strength = x * sigma
    },
    "val_dataloader": {
        "batch_size": 1,
        "num_workers": 1,
        "shuffle": false,
        "val_data_file": "data/preprocessed/val_BSD.h5"
    },
    "logging_info": {
        "epochs_per_val": 1,
        "log_dir": "../trained_models/", //dir to save the model
        "save_epoch": 5 //the model is saved every x epochs
    },
}
```
