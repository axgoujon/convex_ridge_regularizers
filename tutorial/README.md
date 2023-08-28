In this folder we show how to deploy trained CRR-NNs to solve inverse problems.
# Solving Inverse Problems with CRR-NNs
The repository contains the pretrained convex-ridge regularizers (CRR-NNs) corresponding to [paper](https://ieeexplore.ieee.org/document/10223264) (or [open access version](https://arxiv.org/pdf/2211.12461.pdf)).

The details given hereafter are illustrated in the notebnook [howto.ipynb](https://github.com/axgoujon/convex_ridge_regularizers/blob/main/howto.ipynb).

Requirements
--------------
* python >= 3.8
* pytorch >= 1.12
* (optional) CUDA

Pretrained CRR-NNs
--------------
Each of the 14 given regularizers $R_{t,\sigma}$ was trained via a $t$-step denoising task, $t=1, 2, 5, 10, 20, 30, 50$ with noise level $\sigma$, $\sigma=5, 25$.
The performance of $R_{t,\sigma}$ marginally depends on $t$, and tuning this hyperparameter to solve an inverse problem is not necessary in general.

This document gives a few implementation details, more information in the notebook howto.ipynb.

Solving an Inverse Problem
--------------
Given a forward operator $\mathbf{H}$ and some measurements $\mathbf{y}$, CRR-NNs are plugged into the variational problem

<img src="https://latex.codecogs.com/svg.image?\mathrm{argmin}_{\mathbf{x}}&space;\frac{1}{2}\|\mathbf{H}\mathbf{x}&space;-&space;\mathbf{y}\|_2^2&space;&plus;\lambda/\mu&space;R_{t,\sigma}(\mu\mathbf{x})." />

The regularizer $R_{t,\sigma}$ is smooth, and the optimization problem can be solved with any standard gradient-based method, e.g. accelerated gradient-descent. If $\mathbf{x}$ needs to be constrained, e.g. to be positive, the FISTA algorithm is well-suited. For these algorithms the following functions are useful:
- `model.precise_lipschitz_bound()` to compute a bound on the Lipschitz constant of $\nabla R_{t,\sigma}$ and determine an admissible stepsize.
Nb: given $\lambda$ and $\mu$, the used bound should be `lmbd * mu * model.precise_lipschitz_bound()`, i.e. $\lambda$ and $\mu$ are not included into the model.
- `model.grad(x)` to compute $\nabla R_{t,\sigma}(\mathbf{x})$.
Nb: given $\lambda$ and $\mu$, the gradient actually used is `lmbd * model.grad(mu * x)`, i.e. $\lambda$ and $\mu$ are not included into the model.

In addition,
- `model.cost(x)` gives the regularization cost. Again, $\lambda$ and $\mu$ have to be manually added as in `lmbd/mu*model.cost(mu * x)`.

Tuning $\lambda$ and $\mu$
--------------
In the [paper](https://ieeexplore.ieee.org/document/10223264) (or [open access version](https://arxiv.org/pdf/2211.12461.pdf)) and the folder [hyperparameter_tuning](https://github.com/axgoujon/convex_ridge_regularizers/blob/hyperparameter_tuning/), we give a precise routine to tune both hyperparameters.

Although tuning $\mu$ is important, it can be done manually in a coarse manner. Typically, increasing $\mu$ improves the performance, and after some point the results depend only marginally on $\mu$. Typical range for $\mu$ is [1, 200]. Nb: setting $\mu$ to a too large value yields a "less smooth" regularizer and convergence usually becomes slower.


