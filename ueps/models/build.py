import torch
import torch.nn as nn
import os
import yaml

from .recon import ReconFramework

def count_param(model):
    c = sum(p.numel() for p in model.parameters() if p.requires_grad)
    s = "Total learnable parameters: "
    if c <= 1e3:
        s += f"{c}"
    elif c > 1e3 and c <= 1e6:
        s += "{0:.2f} K".format(c/1e3)
    elif c > 1e6 and c <= 1e9:
        s += "{0:.2f} M".format(c/1e6)
    elif c > 1e9:
        s += "{0:.2f} G".format(c/1e9)
    print(s)
    return c, s

def build_model(config=None, device=None, config_path=None, model_path=None, loadweight=False):
    """A group of train methods
    """
    if device is None:
        device = torch.device("cpu")

    if config is None:
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise RuntimeError("config unavailable")

    model = ReconFramework(config).to(device)
    count_param(model)

    if loadweight:
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"Load checkpoint from {model_path}")

    return model

def get_loss(loss_type: str = "mae"):
    if loss_type == "mse":
        loss_fn = nn.MSELoss()

    elif loss_type == "mae":
        loss_fn = nn.L1Loss()

    else:
        raise RuntimeError(f"loss_type {loss_type} is not suppored.")

    return loss_fn
