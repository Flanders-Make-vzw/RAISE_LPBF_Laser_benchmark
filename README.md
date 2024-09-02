RAISE-LPBF-Laser benchmark repo
===
Official repository of the RAISE-LPBF-Laser benchmark.

We hereby provide PyTorch code to load the [dataset](#datasets) as well as the [models](#models) we used as baselines.

More information can be found on our website [makebench.eu](https://www.makebench.eu/benchmark/The%20RAISE-LPBF-Laser%20benchmark).

## Datasets
A collection of PyTorch Datasets can be found in the `dataset.py` file:
- `FramesSP`: extracts frames for training/testing given the filepath to a RAISE-LPBF-Laser HDF5 dataset as downloaded from the website;
- `OneWaySP`: inherits from `FramesSP` to further preprocess the frames for compatibility with video recognition models such as 3DResnet, X3D, MViT, etc.;
- `TwoWaysSP`: inherits from `FramesSP` to further preprocess the frames for compatibility with two-way model SlowFast.

This is just a baseline to demonstrate how to use the data; doubtlessly it is possible to achieve better performance.  Note that the current version of this code expects the v1.0 dataset.
We encourage everyone to submit results of improved models or preprocessing methods to Makebench.eu.

## Models
A collection of models as PyTorch Modules can be found in the `models/` folder:
- `CNN3DResnet`: Hara et Al. (2017). Learning spatio-temporal features with 3D residual networks for action recognition.
- `CNN3DSlowFast`: Feichtenhofer et Al. (2019). SlowFast Networks for Video Recognition.
- `MViT`: Fan et Al. (2021). Multiscale Vision Transformers.
- `Swin3D`: Yang et Al. (2023). Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding.
- `ViViT`: Arnab et Al. (2021). ViViT: A Video Vision Transformer.
- `X3D`: Feichtenhofer et Al. (2020). X3D: Expanding Architectures for Efficient Video Recognition.


## Citation
```bibtex
@article{BLANC2023100161,
title = {Reference dataset and benchmark for reconstructing laser parameters from on-axis video in powder bed fusion of bulk stainless steel},
journal = {Additive Manufacturing Letters},
volume = {7},
pages = {100161},
year = {2023},
issn = {2772-3690},
doi = {https://doi.org/10.1016/j.addlet.2023.100161},
url = {https://www.sciencedirect.com/science/article/pii/S2772369023000427},
author = {Cyril Blanc and Ayyoub Ahar and Kurt {De Grave}},
keywords = {Selective laser melting, Stainless steel, On-axis camera, Dataset, Machine learning, Monitoring}
}
```
