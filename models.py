import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Union, Optional
import warnings

from utils import rdelattr, rsetattr


class UNet(nn.Module):
    """
    Standard Unet

    Args:
        int_c (int): input channels
        out_c (int): output channels
        n_down (int): number of downsampling steps
        n_fix (int): number of convolutional layers at each resolution
        C (int): number of channels at max resolution
        Instance_norm (bool): Weither to use instance norm or batchnorm
        n_dim (int): dimension of input tensor (2 or 3)
    """
    def __init__(self, in_c: int, out_c: int, n_down: int, n_fix: int, C: int, Instance_norm: bool=True, n_dim: int=3) -> None:
        super(UNet, self).__init__()
        
        self.n_down = n_down
        self.n_fix = n_fix
        self.C = C
        self.IN = Instance_norm
        self.in_c = in_c
        self.out_c = out_c
        self.n_dim = n_dim

        if n_dim not in [2, 3]:
            raise ValueError(f"n_dim should be 2 or 3, got {n_dim}.")

        conv = nn.Conv3d if n_dim == 3 else nn.Conv2d
        transposed_conv = nn.ConvTranspose3d if n_dim == 3 else nn.ConvTranspose2d
        instance_norm = nn.InstanceNorm3d if n_dim == 3 else nn.InstanceNorm2d
        batch_norm = nn.BatchNorm3d if n_dim == 3 else nn.BatchNorm2d
        norm_layer = instance_norm if self.IN else batch_norm

        self.conv_init = conv(in_c, C, 3, 1, 1)  
        self.act_init = nn.ReLU()
        self.norm_init = norm_layer(C, affine=True)

        for l in range(n_fix):
            setattr(self, "conv_0_" + str(l), conv(C, C, 3, 1, 1))
            setattr(self, "act_0_" + str(l), nn.ReLU())
            setattr(self, "norm_0_" + str(l), norm_layer(C, affine=True))
        for lvl in range(n_down):
            setattr(self, "down_" + str(lvl), conv(2**(lvl) * C, 2**(lvl + 1) * C, 3, 2, 1))
            setattr(self, "down_act_" + str(lvl), nn.ReLU())
            setattr(self, "down_norm_" + str(lvl), norm_layer(2**(lvl + 1) * C, affine=True))
            for l in range(n_fix):
                setattr(self, f"conv_{lvl+1}_{l}", conv(2**(lvl + 1) * C, 2**(lvl + 1) * C, 3, 1, 1))
                setattr(self, f"act_{lvl+1}_{l}", nn.ReLU())
                setattr(self, f"norm_{lvl+1}_{l}", norm_layer(2**(lvl + 1) * C, affine=True))
        for lvl in range(n_down):
            setattr(self, "up_" + str(lvl), transposed_conv(2**(lvl + 1) * C, 2**(lvl) * C, 4, 2, 1))
            setattr(self, "up_act_" + str(lvl), nn.ReLU())
            setattr(self, "up_norm_" + str(lvl), norm_layer(2**(lvl) * C, affine=True))
            for l in range(n_fix):
                if l == 0:
                    setattr(self, "dec_conv_" + str(lvl) + "_0", conv(2**(lvl + 1) * C, 2**(lvl) * C, 3, 1, 1))
                else:
                    setattr(self, "dec_conv_" + str(lvl) + "_" + str(l), conv(2**(lvl) * C, 2**(lvl) * C, 3, 1, 1))
                setattr(self, "dec_act_" + str(lvl) + "_" + str(l), nn.ReLU())
                setattr(self, "dec_norm_" + str(lvl) + "_" + str(l), norm_layer(2**(lvl) * C, affine=True))
        self.conv_final = conv(C, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = self.conv_init(x)
        x = self.act_init(x)
        x = self.norm_init(x)
        L = []
        for l in range(self.n_fix):
            x = getattr(self, "conv_0_" + str(l))(x)
            x = getattr(self, "act_0_" + str(l))(x)
            x = getattr(self, "norm_0_" + str(l))(x)
        L.append(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_" + str(lvl))(x)
            x = getattr(self, "down_act_" + str(lvl))(x)
            x = getattr(self, "down_norm_" + str(lvl))(x)
            for l in range(self.n_fix):
                x = getattr(self, f"conv_{lvl+1}_{l}")(x)
                x = getattr(self, f"act_{lvl+1}_{l}")(x)
                x = getattr(self, f"norm_{lvl+1}_{l}")(x)
            L.append(x)
        for lvl in range(self.n_down - 1, -1, -1):
            x = getattr(self, "up_" + str(lvl))(x)
            x = getattr(self, "up_act_" + str(lvl))(x)
            x = getattr(self, "up_norm_" + str(lvl))(x)
            x = torch.cat([x, L[lvl]], dim=1)
            for l in range(self.n_fix):
                x = getattr(self, "dec_conv_" + str(lvl) + "_" + str(l))(x)
                x = getattr(self, "dec_act_" + str(lvl) + "_" + str(l))(x)
                x = getattr(self, "dec_norm_" + str(lvl) + "_" + str(l))(x)
        x = self.conv_final(x)
        return x
    

class HyperModel(nn.Module):
    """
    Model that dynamically creates weights for a neural network conditioned on a vector-valued variable such as image spacing, patient age, contrast phase, etc..
    The forward pass is performed by generating the main model's weights from the conditioning variable and then performing a forward pass with the main model.
    See the `HyperNetworks paper <https://arxiv.org/abs/1609.09106v4>` for more details.

    .. note::
        The model can only be traced with a single conditioning variable. The traced model can be used for batched inference with a single conditioning variable.

    Args:
        hypernetwork_layers (list): list of sizes of the hypernetwork hidden layers, starts with the dimension of the conditioning variable
                                    and then corresponds to hidden layer widths
        primary_model (nn.Module): primary model for which the parameters will be predicted. Any torch model can be specified.
    """

    def __init__(self,
                 hypernetwork_layers: list[int],
                 primary_model: nn.Module):

        if not hypernetwork_layers:
            raise ValueError("hypernetwork_layers must contain at least input dimensions")

        super(HyperModel, self).__init__()

        # Create HyperNetwork
        self.hypernetwork = nn.Sequential(*[
            nn.Sequential(nn.Linear(hypernetwork_layers[i], hypernetwork_layers[i + 1]), nn.ReLU())
            for i in range(len(hypernetwork_layers) - 1)
        ]) if len(hypernetwork_layers)>1 else nn.Identity()
        self.conditioning_dim = hypernetwork_layers[0]
        self.hidden_size = hypernetwork_layers[-1]

        self.model = primary_model

        named_parameters = self.model.state_dict()
        self.param_shapes = {}
        heads = {}

        for name, param in named_parameters.items():
            self.param_shapes[name] = param.size()
            n_elements = param.shape.numel()
            # Remove dots in the name because ModuleDict does not support it
            if ";" in name:
                raise ValueError(
                    f"Layer {name}'s name contains semicolons, this is not supported. Change the name of that layer.")
            heads[name.replace(".", ";")] = nn.Linear(self.hidden_size, n_elements)
            # set the bias terms to predict the current parameters for zero conditioning
            heads[name.replace(".", ";")].bias.data = param.view(-1)
            # Need to delete the parameters to be able to use the setter with predicted tensors in the forward (see line 172)
            rdelattr(self.model, name)

        # Register a dummy parameter for the create_dummy_data method to work
        self.model.dummy_param = nn.Parameter(torch.rand(1, 1))

        self.heads = nn.ModuleDict(heads)

    def create_conditional_model(self, hypernetwork_base_prediction: torch.Tensor) -> nn.Module:
        """
        Re-create weights in the sub-modules of the main model using the hypernetwork's prediction and the layer heads

        Args:
            hypernetwork_base_prediction (torch.Tensor): The output of the hypernetwork for a specific conditioning sample.
        """
        for name, layer in self.heads.items():
            # Custom activation used as final activation of the hypernetwork to constraint the predicted weigts within the sphere of infinit norm 5
            param = F.tanh(layer(hypernetwork_base_prediction)) * 5
            rsetattr(self.model,
                     name.replace(";", "."),
                     torch.reshape(param, self.param_shapes[name.replace(";", ".")]))
        return self.model

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):  # type: ignore[override]
        """
        Perform a forward pass on the main model with the weights set by the hypernetwork's prediction for each element in the batch. If the model is being traced, the conditioning is interpreted as constant and sets up the same network parameters for the entire batch.
        
        Args:
            x (torch.Tensor): batch of input tensors for the primary network
            conditioning (torch.Tensor): batch of conditioning variable from which the primary network's weights will be predicted
        """
        if torch.jit.is_tracing():
            with warnings.catch_warnings():  # prevent a warning from pytorch when tracing
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                if conditioning.size(0) != 1:
                    raise ValueError("Tracing hyper-model with conditioning size > 1 is not supported).")
            hidden = self.hypernetwork(conditioning[0:1])  # selection [0:1] needed for tracing
            model = self.create_conditional_model(hidden)
            return model(x)

        # Process conditioning values one by one
        result = []
        for i in range(len(conditioning)):
            hidden = self.hypernetwork(conditioning[i:i + 1])
            model = self.create_conditional_model(hidden)
            result.append(model(x[i:i + 1]))

        return torch.cat(result, dim=0)
