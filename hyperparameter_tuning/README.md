Given a score function, the script [validate_coarse_to_fine.py](https://github.com/axgoujon/convex_ridge_regularizers/blob/main/validate_coarse_to_fine.py) allows one to tune two hyperparameters with the simple coarse-to-fine approach given in the [paper](https://ieeexplore.ieee.org/document/10223264) (or [open access version](https://arxiv.org/pdf/2211.12461.pdf)).


Requirements
--------------
* python >= 3.8
* pandas
* numpy

How to
--------------
The goal is to tune two parameters (p1,p2), for e.g. $(\lambda,\mu)$.

 **Requirements**
 The routine requires a score function `score(p1, p2)`. This function takes the hyperparameters as input and should return a performance metric. Typically, this function loops over the validation set, solve the inverse problem for each image, then return the average performance in the form `(psnr, ssim, niter)`.
 
 **Nb 1:** as implemented, the optimization is based only on the PSNR, the SSIM and the number of iterations to converge are used only for logging. Hence their value does not modify the outcome of the algorithm and you could put any other metric.

**Nb 2:** if optimizing a single parameter (```freeze_p2=True```), `score(p1)` is expected to only take a single input.

 **Usage**
 ```python
 # initialize the validation process
 validator = ValidateCoarseToFine(score, dir_name="./", exp_name="CsMRI_Mask1", p1_init=0.1, p2_init=10, freeze_p2=False)
# run the validation
validator.run()
```
**Output**
Each time the score function is called, the results are stored in a local database, namely a simple *.csv* file. The process is stopped when the grid size is sufficiently small. Then, one can identify the more promising parameters from the *.csv*.

The *.csv* also allows to skip the calls to the score function for parameters that have already been used.
 
**Tuning one parameter**
The routine can be used to tune a single parameter by setting `freeze_p2=True`.

**Validation set**
For the experiments presented in the [paper](https://ieeexplore.ieee.org/document/10223264) (or [open access version](https://arxiv.org/pdf/2211.12461.pdf)), it was noticed that a small validation set (<=10 well chosen samples) suffices to generalize well. Hence the tuning phase is rather fast.