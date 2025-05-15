# HyperSORT
Code for the MICCAI 2025 submission "HyperSORT: Self-Organising Robust Training with hyper-networks"

# Overview

HyperSORT leverages hyper-networks to learn a meaningful latent space parameterizing and discovering the different annotation biases within a training set. The hyper-network maps annotation style latent vectors from the latent space to UNet parameters performing the segmentation task according to the corresponding annotation bias.

<!-- ![HyperSORT overview](images/Graphical_abstract.png "Overview of HyperSORT") -->
<img src="images/Graphical_abstract.png" alt="Overview of HyperSORT" width="1000"/>

# Guidelines

## 1. Create Python environment

Create a Python virtual environment and install the requirements.

Please follow the [Pytorch installation guidelines for your specific setting](https://pytorch.org/). 

Please set-up the free ImFusion license key as specified [in the ImFusion python SDK documentation](https://docs.imfusion.com/python/installing.html#license-activation).

## 2. Download dataset and create data file

In HyperSORT, experiments were conducted on the [AMOS dataset](https://zenodo.org/records/7262581) and on the [TotalSegmentator CT dataset](https://zenodo.org/records/6802614) (the corrected V2 version is also available [here](https://zenodo.org/records/8367088)). 
Yet, HyperSORT can be used on any dataset. Feel free to apply it to any dataset you are interested in.

Once the training dataset is downloaded, create a data file. 
A data file is a text file where each line is a tab-separated list of files for a training case. 

Templates for such data files can be found in `data_files/data_list_TS.txt` and `data_files/data_list_AMOS.txt`.

## 3. Create a training yaml config

Create a training yaml config parameterizing your experiments. The used configs for our experiments can be found in: `configs/AMOS.yaml` and `configs/TS.yaml`.

A config yaml file must specify the following sections:

**OutputPath**: Specifying the folder where the experiment training plots and results will be saved.

**DataFile**: The path to the created data file specifying the files constituting the training set to learn the annotation style latent space from.

**Pipeline**: A list of preprocessing operations to be applied to the image. We used the ImFusion SDK to preprocess images. Documentation on available operations can be found [here](https://docs.imfusion.com/python/ml_op_bindings.html). Each preprocessing operation is specified as:

```
    - OperationName: # without the suffix "Operation"
        OperationParameterName1: OperationParameterValue1
        ... 
```

**ModelConfig**: Specify HyperSORT's model parameters
    
- HyperNetworkLayers is a list starting with the latent space dimension followed by the hyper-network hidden layers dimensions
    
- The UNet section specifies the UNet archtiecture's parameters

## 4. Launch HyperSORT training

Run:

`python main.py -c path_to_config_file.yaml`





