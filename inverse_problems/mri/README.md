# Single- and multi-coil MRI

**Preprocessed data:** download the  data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8302121.svg)](https://doi.org/10.5281/zenodo.8302121)
, unzip it and put it in the data folder under the name `data_sets.

The data contains validation (aka calibration) and test sets with:
- subsampling cartesian masks,
- sensitivity masks,
- ground truth image,
- measurements,
for the various settings explored: single- and multi-coil MRI, various acceleration rates (2, 4, and 8), synthetic noise and different image type (fat suppression or not).

For completeness we also put the code used to generate the preprocessed data from the raw data. (Nb need the bart library to generate the datasets, not needed with the data link).