# A Neural-Network-Based Convex Regularizer for Inverse Problems
**New** *Check the extension of CRR-NNs to weakly-convex regularizers [preprint](https://arxiv.org/abs/2308.10542)/[repo](https://github.com/axgoujon/weakly_convex_ridge_regularizer). Remark: the latter repository also includes a multi-noise-level deep-equilibrium training of CRR-NNs with a slightly simpler procedure (shared activations + no spline regularization), with same denoising performance.*
#

Implementation of the CRR-NNs as presented in this [paper](https://ieeexplore.ieee.org/document/10223264) (or [open access version](https://arxiv.org/pdf/2211.12461.pdf)). For any bug report/question/help needed please [contact us](mailto:alexis.goujon@epfl.ch).

*Nb: detailed information is provided in the README.me of each folder.*

The repository is organized as follows:
- **trained_models:** contains a few instances of trained CRR-NNs.
- **tutorial:** to understand and deploy trained CRR-NNs for solving inverse problems.
- **inverse_problems:** contains the MRI and CT experiments + generic utils for solving inverse problems and for the automated validation and testing.
- **denoising:** to reproduce the denoising results on BSD68 with the provided trained models.
- **training:** to reproduce training.
- **under_the_hood:** to visualize the NNs (filters and activations).
- **models:** contains the spline modules, the CRR-NNs class, some optimization schemes (t-step denoiser and accelerated gradient descent)...
- **hyperparameter_tuning:** helpers and discussion on the tuning of $\lambda$ and $\mu$ for solving inverse problems. See **inverse_problems** folder for usage. 

Some requirements (depending on what you use)
--------------
* python >= 3.8
* pytorch >= 1.12
* (optional) CUDA
* numpy
* pandas
* tensorboard
* matplotlib
* Pillow

For the CT experiments:
* astra-toolbox

For the MRI expeirements:
* fastmri
* bart
