This folder is dedicated to denoising. Given a CRR-NN trained as a multi-gradient-step denoiser, it yields two different denoisers.

# Proximal denoiser
There, the denoised image is the solution of a convex minimization problem. The minimization is performed with an accelerated gradient descent with restart and a very small tolerance ($10^{-6}$) to ensure full convergence---in practice, depending on the application, early stopping can be used to boost performance and efficiency by decreasing the tolerance (dig into the code).
## Validation
To turn the trained multi-gradient-step denoiser into a proximal denoiser it is necessary to tune $\lambda$ (major impact) and $\mu$ (secondary impact). This can be done manually, but we also provide the validation routine that we have used for fine tuning, more info in the /hyperparameters_tuning/ folder. For e.g.

```console
:~/convex_ridge_regularizers/denoising$ python validation_proximal_denoisers.py
``` 


**Nb:** for the validation you can explore different models, specified by $t$ and $\sigma_{\mathrm{train}}$ used at trained, and various noise levels $\sigma_{\mathrm{test}}$. Modify directly the code to control this.

The validation results for all value of $(\lambda, \mu)$ explored are stored in a .csv dataframe, which is read to set the hyperparameters for testing. For testing on BSD68

```console
:~/convex_ridge_regularizers/denoising$ python test_proximal_denoisers.py --noise_level_train 5 --noise_level_test 25 --t 1
```
Parameters:
- --noise_level_train (used to select the model, the model given were trained with 5 and 25)
- --noise_level_test any value, **for which the validation routine was run** (we provide for 5 and 25 for both training noise levels 5 and 25)
- --t (used to select the model, the model given were trained with t=1, 2, 5, 10, 20, 30 50)

**To test on noise levels $\in\{1, 2, \ldots, 30\}$ run**

```console
:~/convex_ridge_regularizers/denoising$ python test_proximal_denoisers.py --noise_level_train 5 --noise_level_test 25 --t 10
```
since we provide validation data for $t=10$ and noise_level_train=5.
# t-step denoisers
These denoisers are made of $t$ iterations of gradient descent and directly correspond to what was done at training. Hence they can be directly used for denoising.

```console
:~/convex_ridge_regularizers/denoising$ python test_t-step_denoisers.py --noise_level 5 --t 1
```
*(here the noise_level relates both to the training and testing noise level)*

**Nb:** 
- the Lipschitz constant of the gradient of the model is updated before inference.
- on the stepsize rule used for the partial gradient descent scheme.
    - this rule should be consistent for training and testing
    - whether the denoiser is **provably non-expansive** or not depends on this rule, it is subtle => see paper for details.
    (nb: by default the $t$-step denoiser is non-expansive for $t=1, +\infty$ but not necessarily for $1<t<+\infty$. If non-expansive denoiser is needed for $1<t<+\infty$, you need to retrain and before edit the `tStepDenoiser` routine in `models/utils.py`, the details are given as comments in the code.)


# Computational Speed
If speed is important, you have a few options. It is possible to **prune** for inference a trained CRR-NN with this line of code (put just after the model is load):
```python
model.prune(prune_filters, collapse_filters, change_splines_to_clip, tol)
```



- *prune_filters:* if true, the tuples (filter, activation) with marginal importance (marginal being quantified via the tol parameter) are removed from the model
- *collapse_filters:* the filters are by default made of composition of 2 convolutional layers. This yield a linear operator but not exactly a convolution (this is it only because of boundary effects). This option allows one to replace the 2 convolutions by one with the equivalent kernel, which boosts the efficiency. For denoising we noticed a dropped in performance of 0.05/0.1 dB, but in the inverse problems explored the difference was not significant.
- *change_splines_to_clip:* allows one to express the splines as sum of two ReLUs. The implementation is quite naive and assumes some clip shape of the activation =>could possibly alter the performances so pay attention!


For proximal denoisers: large values of $\mu$ makes the activation "almost" non $C^1$, and accelerated gradient descent is not very efficient anymore => in the validation routine ```ValidateCoarseToFine``` you can force $\mu$ to remain lower than a threshold specified by `p2_max`, e.g.

```python
ValidateCoarseToFine(score, dir_name, exp_name, p1_init=1, p2_init=1, p2_max=5)
```

or freeze it to some small value e.g. 5
```python
ValidateCoarseToFine(score, dir_name, exp_name, p1_init=1, p2_init=5, freeze_p2=True)
```

Otherwise, you can explore alternative optimization routine better tailored to large $\mu$, e.g. with the dual problem.