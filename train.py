import utils
import numpy as np

import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import dice_loss, cross_entropy_loss


def train(model, dataloader, latent_size, path_res, max_epoch=10000):
    """
    Training loop for HyperSORT

    model (nn.Module): HyperSORT model (hyper-UNet)
    dataloader (DataLoader): torch data loader, should provide the input image, the label map and a unique identifier for the image/label pair
    latent_size (int): size of the annotation style's latent space
    path_res (str): path for the result experiment folder
    """

    # Log to track loss values
    log = {"Train": defaultdict(list)}

    hypernetwork_optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    latent_dict = {}  # Dictionary storing latent vectors (created on the fly)
    latent_optimizer_dict = {} # Dictionary storing latent vector specific optimizers (created on the fly)

    epoch = 0
    while True:
        for n, batch in tqdm(enumerate(dataloader)):
            
            x = batch["image"].cuda()
            y = batch["label"].cuda()
            y = F.one_hot((y[:, 0]>0).long(), num_classes=2).permute(0, -1, *list(range(1, len(y.size())-1))).float()

            if x.size(0)>1:
                raise ValueError("Current implementation does not support batch size larger than 1.")
            
            identifier = batch["data_identifier"][0].item()

            if identifier not in latent_dict:
                # Create latent vector entry
                latent_dict[identifier] = torch.zeros((1, latent_size), requires_grad=True, device="cuda")
                torch.nn.init.normal_(latent_dict[identifier], std=0.05)
                latent_params = (latent_dict[identifier],)
                latent_optimizer_dict[identifier] = torch.optim.Adam(
                    params=latent_params, betas=(0.7, 0.9), lr=0.001
                )

            latent = latent_dict[identifier]
            latent_optimizer = latent_optimizer_dict[identifier]
            
            latent_optimizer.zero_grad()
            hypernetwork_optimizer.zero_grad()

            prediction = model(x, latent)
            prediction_prob = F.softmax(prediction, dim=1)
            
            latent_reg = torch.norm(latent, p=1)
            ce = cross_entropy_loss(prediction, y).mean()
            dice = dice_loss(prediction_prob, y).mean()
            prediction_loss = ce + dice
            loss = prediction_loss + latent_reg

            # Log losses
            log["Train"]["norm"].append(latent_reg.item())
            log["Train"]["Dice loss"].append(dice.item())
            log["Train"]["CE loss"].append(ce.item())
            log["Train"]["Seg loss"].append(prediction_loss.item())
            log["Train"]["log Seg loss"].append(np.log(prediction_loss.item() + 1e-8))

            loss.backward()
            latent_optimizer.step()
            hypernetwork_optimizer.step()

            # For ImFusionSuite users, use the first line, saving images to .imf format for better vizualization
            if epoch % 100 == 0 and n<5:
                # utils.save_cases([x], [prediction], [y], os.path.join(path_res, "batches", f"batch-{epoch}-{n}.imf"))
                utils.save_cases([x], [prediction, y], [], os.path.join(path_res, "batches", f"batch-{epoch}-{n}.nii.gz"))

        # Create plots for training monitoring
        utils.plot_loss(log, os.path.join(path_res, "loss_plot.png"), ma=10)
        utils.create_interactive_scatter(
            latent_dict, os.path.join(path_res, "plots", "interactive", f"scatter-{epoch}.html")
        )
        utils.create_scatter(
            latent_dict, os.path.join(path_res, "plots", "scatter", f"scatter-{epoch}.png")
        )
        epoch += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(path_res, "model_w.pt"))
            torch.save(latent_dict, os.path.join(path_res, "latent_dict.pt"))

        if epoch>max_epoch:
            return
