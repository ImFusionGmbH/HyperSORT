import argparse
import os
import shutil
from torch.utils.data import DataLoader
import yaml

from train import train 
from data import ImFusionIODataset
import models
import utils



parser = argparse.ArgumentParser("Hyper Boost")
parser.add_argument("-c", "--config", type=str, required=True)

args = parser.parse_args()


with open(args.config, "r") as f:
    config = yaml.safe_load(f)

path_res = config.get("OutputPath", "")

# Create output repositories
utils.makedir(path_res)
utils.makedir(os.path.join(path_res, "batches"))
utils.makedir(os.path.join(path_res, "training_files"))
path_plots = os.path.join(path_res, "plots")
utils.makedir(path_plots)
utils.makedir(os.path.join(path_plots, "interactive"))
utils.makedir(os.path.join(path_plots, "scatter"))

# Copy files for reproducibility
files_to_copy = [
    args.config,
    "data.py",
    "models.py",
    "main.py",
    "train.py",
    "utils.py",
]
for file_to_copy in files_to_copy:
    shutil.copyfile(file_to_copy, os.path.join(path_res, "training_files", os.path.basename(file_to_copy)))


# Model creation
model_config = config.get("ModelConfig", {})
unet_config = model_config.get("UNet", {})
primary_model = models.UNet(**unet_config)
hypernetwork_layers = model_config.get("HyperNetworkLayers")
model = models.HyperModel(hypernetwork_layers, primary_model).cuda()


# Dataset creation
data_file = config["DataFile"]
pipeline = config["Pipeline"]
fields = [("image", 0, 0), ("label", 1, 0)]
dataset = ImFusionIODataset(data_file, fields, pipeline)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# Training
latent_size = hypernetwork_layers[0]
train(model, dataloader, latent_size, path_res)

