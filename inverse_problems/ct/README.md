# Data

## Downloads
You have two options.
### (faster) Preprocessed data
Download the  data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8302121.svg)](https://doi.org/10.5281/zenodo.8302121)
, unzip it and put it in the data folder under the name `data_sets.

The data contains validation (aka calibration) and test sets with:
- ground truth image,
- FBP reconstruction,
- measurements,
for the various settings explored (3 noise levels in the measurements).


### Raw data
For completeness we also give the routine used to get the preprocessed data.

We use the same data and setup as [link](https://github.com/Subhadip-1/data_driven_convex_regularization/):
The phantoms are available (as .npy files) here: [link](https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD?usp=sharing). Download the .zip file containing the phantoms, unzip, and put inside the cloned directory.

#### Processing
To generate the validation and test sets run
```
python make_data_sets.py
```