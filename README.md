RAISE-LPBF-Laser benchmark repo
===
Official repository of the RAISE-LPBF-Laser benchmark.

We hereby provide Pytorch-compatible code to interface our [datasets](#datasets) as well as the [models](#models) we used as baselines.

More information can be found on our website [makebench.eu](https://www.makebench.eu/benchmark/The%20RAISE-LPBF-Laser%20benchmark).

## Datasets
A collection of Pytorch Datasets can be found in the `dataset.py` file:
- `FramesSP`: extracts and process data for training/testing given the filepath to a RAISE-LPBF-Laser HDF5 dataset;
- `OneWaySP`: inherits from `FramesSP` to process the frames even more for compatibility with one-way models such as 3DResnet, X3D, MViT, etc.;
- `TwoWaysSP`: inherits from `FramesSP` to process the frames even more for compatibility with one-way models such as SlowFast.


## Models
A collection of models as Pytorch Modules can be found in the `models/` folder:
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
keywords = {Selective laser melting, Stainless steel, On-axis camera, Dataset, Machine learning, Monitoring},
abstract = {We present RAISE-LPBF, a large dataset on the effect of laser power and laser dot speed in powder bed fusion (LPBF) of 316L stainless steel bulk material, monitored by on-axis 20k FPS video. Both process parameters are independently sampled for each scan line from a continuous distribution, so interactions of different parameter choices can be investigated. The data can be used to derive statistical properties of LPBF, as well as to build anomaly detectors. We provide example source code for loading the data, baseline machine learning models and results, and a public benchmark to evaluate predictive models.}
}
```