import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from typing import Optional, Union, Any, Mapping

import imfusion
from imfusion import SharedImageSet


#################### Loss functions


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice loss

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor
    """
    axes = list(range(2, len(pred.size())))
    return 1 - 2*(pred*target).sum(dim=axes) / (pred.sum(dim=axes)+target.sum(dim=axes)+1e-8)


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross Entropy loss

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor
    """
    return F.cross_entropy(pred, target)


#################### Plot functions


def plot_loss(loss_dict: dict, path: str, ma: int=100) -> None:
    """
    Plot training log

    Args:
    loss_dict (dict): log object in the form a dictionary as follow 
                        {
                            phase: {
                                loss_name : [list_values]
                            }
                        }
    path (str): path where the plot image file will be saved
    ma (int): moving average window size
    """
    n_losses_train = len([k for k in loss_dict.get("Train", {}).keys()])
    n_losses_val = len([k for k in loss_dict.get("Val", {}).keys()])
    n_losses = max(n_losses_train, n_losses_val)
    fig, axes = plt.subplots(2 if (n_losses_val>0) else 1,
                                 n_losses,
                                 figsize=(15 * n_losses, 20))
    if n_losses_val==0:
        phases = ["Train"]
        axes = [axes]
    else:
        phases = ["Train", "Val"]

    for i, phase in enumerate(phases):
        for idx, loss_name in enumerate(loss_dict.get(phase, {})):
            data = moving_average(loss_dict[phase][loss_name], ma)
            axes[i][idx].plot(np.arange(len(data)),
                                data)
            axes[i][idx].set_title(phase + ' ' + loss_name)

    fig.savefig(path, bbox_inches='tight')
    plt.clf()


def create_scatter(latent_dict, path, colors=None):
    """
    Create a scatter plot from the latent dictionary to visualize latent vectors distribution

    Args:
        latent_dict (dict): Dictionary containing the latent vectors
        path (str): path where the plot image file will be saved
        colors (list): optional list of colors associated to each vector in the latent dictionary
    """

    X = [x[0][0].item() for x in latent_dict.values()]
    Y = [x[0][1].item() for x in latent_dict.values()]
    plt.clf()
    plt.scatter(X, Y, c=colors)
    plt.savefig(path)


def create_interactive_scatter(latent_dict, path, title="Interactive Scatter Plot", colors=None):
    """
    Create an interactive scatter plot (html) from the latent dictionary to visualize latent vectors distribution

    Args:
        latent_dict (dict): Dictionary containing the latent vectors
        path (str): path where the plot html file will be saved
        title (str): title written at the top of the interactive plot
        colors (list): optional list of colors associated to each vector in the latent dictionary
    """
    x = [x[0][0].item() for x in latent_dict.values()]
    y = [x[0][1].item() for x in latent_dict.values()]
    labels = list(latent_dict.keys())
    
    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        mode='markers',  # Show points as markers
        text=labels,    # Text to display on hover
        hovertemplate=
        '<b>%{text}</b><br>' +  # Bold label
        'x: %{x}<br>' +         # x-value
        'y: %{y}<br>' +         # y-value
        '<extra></extra>',      # Remove trace info
        marker=dict(
            size=8,
            opacity=0.8,
            color=colors
        )
    )])

    fig.update_layout(title=title,
                      xaxis_title="X-axis",
                      yaxis_title="Y-axis",
                      hovermode="closest") #optimize hover behaviour
    fig.write_html(path)
    return fig


#################### helper functions


def makedir(dir: str) -> None:
    """
    Create a directory if it does not exist

    Args:
        dir (str): directory to be created
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)


def save_cases(im: list[torch.Tensor], pred: list[torch.Tensor], lbl: list[torch.Tensor], path: str):
    """
    Helper function to save training or validation batch

    Args:
        im (torch.tensor): Image(s) to be saved
        pred (torch.tensor): Prediction(s) to be saved
        lbl (torch.tensor): Label(s) to be saved
        path (str): path to the training/validation file
    """
    if not (path.endswith(".imf") or path.endswith(".nii.gz")):
        raise ValueError("Only to imf or nii.gz format is supported.")
    axes_perm = (0, 2, 3, 4, 1) if len(im[0].size())==5 else (0, 2, 3, 1)
    sis = []
    for im_ in im:
        sis.append(SharedImageSet(im_.permute(*axes_perm).detach().cpu().numpy()))
    for pred_ in pred:
        sis.append(SharedImageSet(torch.argmax(pred_, dim=1, keepdims=True).permute(*axes_perm).detach().cpu().numpy().astype(np.uint8)))
    for lbl_ in lbl:
        sis.append(SharedImageSet(lbl_.permute(*axes_perm).detach().cpu().numpy().astype(np.uint8)))

    if path.endswith(".imf"):
        imfusion.save(sis, path)
    elif path.endswith(".nii.gz"):
        for i, sis_to_save in enumerate(sis):
            imfusion.save(sis_to_save, path.replace(".nii.gz", f"_{i}.nii.gz"))


def moving_average(data: Union[list, np.ndarray], k: int) -> Union[list, np.ndarray]:
    """
    computes a moving average of the data with a rolling window of size k

    Args:
        data (np.ndarray): data to be running averaged
        k (int): size of the rolling window
    """
    if len(data)<k+1:
        return data
    else:
        return [np.mean(data[i:i+k]) for i in range(len(data) - k)]


#: Dummy Singleton to specify that no default is set.
NO_DEFAULT = type("NO_DEFAULT", (object, ), {})

def rgetattr(obj: Any, attr: str, default: Any = NO_DEFAULT, dict_before_attribute: bool = False) -> Any:
    """
    Recursive variant of getattr that allows to obtain nested attributes (e.g. ``subobject.subsubobject.attr``).
    Dictionary keys will also be checked and can be specified as an attribute in the nested attribute chain.
    Attributes will take precedence over dictionary keys during look-up.

    Args:
        obj (Any): Object from which to retrieve attr.
        attr (str): String specifying the attr name, can be simple or nested (e.g. ``subobj.another_subobj.attr``).
        default (Any): Return value if attr is not found, default: MISSING (will raise if attribute does not exist).
        dict_before_attribute (bool): If True, give precedence to dictionary over attributes during keys look-up
    """

    def __rgetattr(obj: Any, attr: str, default: Any) -> Any:

        if not attr:
            return obj

        inner = None

        # check if the next attribute is present on the current object
        next_attr, *rest = attr.split(".")
        if hasattr(obj, next_attr) or (isinstance(obj, Mapping) and next_attr in obj):

            # since the check above passed we are guaranteed to be able to get the attribute or dict entry
            # attributes will take precedence over dict entries in this implementation.
            try:
                inner = obj[next_attr] if dict_before_attribute else getattr(obj, next_attr)
            except AttributeError:
                inner = getattr(obj, next_attr) if dict_before_attribute else obj[next_attr]

        # check if we are dealing with a list
        if isinstance(obj, list) and next_attr.isdigit():
            inner = obj[int(next_attr)]

        if inner is not None:
            # recurse until we exhaust the attribute chain
            if rest:
                return __rgetattr(inner, ".".join(rest), default)

            # nothing more to look up, return the value of the attribute
            else:
                return inner

        # attribute not found
        else:
            if default is NO_DEFAULT:
                raise AttributeError
            return default

    try:
        return __rgetattr(obj, attr, default)
    except AttributeError:
        raise AttributeError(f"No attribute {attr} in {obj}")


def rsetattr(obj: Any, attr: str, value: Optional[Any] = None) -> None:
    """
    Recursive variant of setattr that allows to set nested attributes (e.g. subobject.subsubobject.attr).
    Dictionary keys will also be checked and can be specified as an attribute in the nested attribute chain.

    Args:
        obj (Any): Object for which to set attr.
        attr (str): String specifying the attr name, can be simple or nested "e.g. subobj.another_subobj.attr".
        value (Any): Value to set.
    """
    sub_obj_list = attr.split(".")
    sub_obj = rgetattr(obj, ".".join(sub_obj_list[:-1]))
    setattr(sub_obj, sub_obj_list[-1], value)


def rdelattr(obj: Any, attr: str) -> None:
    """
    Recursive variant of delattr that allows to delete nested attributes (e.g. subobject.subsubobject.attr).
    Dictionary keys will also be checked and can be specified as an attribute in the nested attribute chain.

    Args:
        obj (Any): Object for which to delete attr.
        attr (str): String specifying the attr name, can be simple or nested "e.g. subobj.another_subobj.attr".
    """
    sub_obj_list = attr.split(".")
    sub_obj = rgetattr(obj, ".".join(sub_obj_list[:-1]))
    if isinstance(sub_obj, dict):
        del sub_obj[sub_obj_list[-1]]
    else:
        delattr(sub_obj, sub_obj_list[-1])
