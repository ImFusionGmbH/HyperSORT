# HyperSORT
Code for the MICCAI 2025 submission "HyperSORT: Self-Organising Robust Training with hyper-networks"

# Overview

HyperSORT leverages hyper-networks to learn a meaningful latent space accounting for the different annotation biases within a training set. The hyper-network maps annotation style latent vectors from the latent space to UNet parameters performing the segmentation task with the corresponded biases.

<!-- ![HyperSORT overview](images/Graphical_abstract.png "Overview of HyperSORT") -->
<img src="images/Graphical_abstract.png" alt="Overview of HyperSORT" width="1000"/>

# Guidelines

## 1. Create Python environment

Create a Python virtual environment and install the requirements.

Please follow the [Pytorch installation guidelines for your specific setting](https://pytorch.org/) and set-up the free ImFusion license key as specified [here](https://docs.imfusion.com/python/installing.html#license-activation).

## 2. Download dataset and create data file

In HyperSORT, we conducted experiments on the [AMOS dataset](https://zenodo.org/records/7262581) and on the [TotalSegmentator CT dataset](https://zenodo.org/records/6802614) (the corrected V2 version is also available [here](https://zenodo.org/records/8367088)). Yet, HyperSORT can be used on any dataset so feel free to apply it to whichever dataset you are interested in.

Once your dataset is downloaded, create a data file which is a text file where each line is a tab-separated list of file for a training case. Template format for such data files can be found in `data_files/data_list_TS.txt` and `data_files/data_list_AMOS.txt`.

## 3. Create a training yaml config

Create a training yaml config parameterizing your experiments. The used configs for our experiments can be found in: `configs/AMOS.yaml` and `configs/TS.yaml`.

A config yaml file must contain different sections:

**OutputPath**: Specifying the folder where the experiment training plots and results will be saved.

**DataFile**: The path to the created data file specifying the files constituting the training set to learn the annotation style latent space from.

**Pipeline**: A list of preprocessing operations to be applied to the image. We used the ImFusion SDK to preprocess images. Documentation on available operations can be found [here](https://docs.imfusion.com/python/ml_op_bindings.html). Each preprocessing operation is specified as:

```
    - OperationName: # without the suffix "Operation"
        OperationParameterName1: OperationParameterValue1
        ... 
```

**ModelConfig**: Specify HyperSORT model parameters
    
- HyperNetworkLayers is a list starting with the latent space dimension followed by the hyper-network hidden layer dimensions
    
- The UNet section specifies the UNet archtiecture's parameters

## 4. Launch HyperSORT training

Run:

`python main.py -c path_to_config_file.yaml`





