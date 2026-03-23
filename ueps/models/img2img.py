import torch
import torch.nn as nn
from typing import Tuple
Tensor = torch.Tensor

from .vitsparse import ViTSparse
from .unet import Unet


class Img2Img(nn.Module):
    """
    A group of models which map a 2D image to a 2D image, with optional input
    norm and shape variation handling.

    Args:
        model_type (str): choose model
        input_norm (bool): if True, apply norm(unnorm) before(after) model
        chan_last (bool): if True, channel dim at last
        coildim_type (str): determine how to handle coil dim
        use_res (bool): if True, apply model in a residual way

    Shape:
        Input: [N, Cin, H, H] or [N, M, Cin, H, H]
        Output: [N, Cout, H, H] or [N, M, Cout, H, H]
        N, M, C, H, W stand for: batch, multi-coil, channel, height, width
        only support H = W

    Supported shape variations:
        1) With or without coil dim, specified by coildim_type
           Require model without coil dim:
               "none": without coil dim
               "as_batch": with coil dim, but merged into batch dim
               "as_chan": with coil dim, but merged into channel dim

        2) Channel dim at -3 or -1, specified by chan_last
    """
    def __init__(self, config):
        super().__init__()
        self.model_type = config.get("model_type", "vitsparse")
        self.input_norm = config.get("input_norm", False)
        self.chan_last = config.get("chan_last", True)
        self.coildim_type = config.get("coildim_type", "none")
        self.use_res = config.get("use_res", False)

        assert self.model_type in ["unet", "vitsparse"]
        assert self.coildim_type in ["none", "as_batch", "as_chan"]

        Cin = config.get("Ncin", 2)
        Cout = config.get("Ncout", 2)
        if self.coildim_type == "as_chan":
            Ncoil = config.get("Ncoil", 1)
            Cin = Cin * Ncoil
            Cout = Cout * Ncoil

        if self.input_norm:
            assert Cin == Cout

        if self.model_type == "unet":
            self.net = Unet(
                in_chans = Cin,
                out_chans = Cout,
                chans = config.get("Nf_unet", 256),
                num_pool_layers = config.get("Ndown", 4),
                drop_prob = config.get("dropout_unet", 0.0),
                )

        elif self.model_type == "vitsparse":
            self.net = ViTSparse(
                H = config.get("Ny", 320),
                Cin = Cin,
                Cout = Cout,
                patch_size = config.get("patch_size", 8),
                posemb_type = config.get("posemb_type", "sincos2d"),
                attention_config = config.get("attention_config", "base"),
                width = config.get("width", None),
                heads = config.get("heads", None),
                layers = config.get("layers", None),
                intermediate_size= config.get("intermediate_size", None),
                q_bias= config.get("q_bias", True),
                k_bias= config.get("k_bias", True),
                v_bias= config.get("v_bias", True),
                proj_bias= config.get("proj_bias", True),
                eps= float(config.get("eps", 1e-5)),
                use_qknorm = config.get("use_qknorm", False),
                use_swiglu = config.get("use_swiglu", False),
                use_rope = config.get("use_rope", False),
                use_rmsnorm=config.get("use_rmsnorm", False),
                attn_impl=config.get("attn_impl", "sdpa"),
                attn_types=config.get("attn_types", None),
                sliding_window=config.get("sliding_window", None),
                chunk_size=config.get("chunk_size", None),
                )

    def shape_prep(self, x: Tensor) -> Tuple[Tensor, torch.Size]:
        if self.coildim_type == "none":
            # without multi-coil dim
            x = x.permute(0, 3, 1, 2).contiguous() if self.chan_last else x
            shape_ori = x.shape # [N, C, H, H]

        else:
            # with multi-coil dim
            x = x.permute(0, 1, 4, 2, 3).contiguous() if self.chan_last else x
            shape_ori = x.shape # [N, M, C, H, H]
            N, M, C, H, W = shape_ori

            if self.coildim_type == "as_batch":
                x = x.reshape(shape=(N*M, C, H, W))
            elif self.coildim_type == "as_chan":
                x = x.reshape(shape=(N, M*C, H, W))

        return x, shape_ori

    def shape_back(self, x: Tensor, shape_ori: torch.Size) -> Tensor:
        if self.coildim_type == "none":
            x = x.permute(0, 2, 3, 1).contiguous() if self.chan_last else x

        elif self.coildim_type in ["as_batch", "as_chan"]:
            N, M, C, H, W = shape_ori
            x = x.reshape(shape=(N, M, -1, H, W))
            x = x.permute(0, 1, 3, 4, 2).contiguous() if self.chan_last else x

        return x

    def norm_net_unnorm(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / std
        x = self.net(x)
        x = x * std + mean
        return x

    def forward(self, x: Tensor) -> Tensor:
        x, shape_ori = self.shape_prep(x)
        x1 = self.norm_net_unnorm(x) if self.input_norm else self.net(x)
        x = x + x1 if self.use_res else x1
        x = self.shape_back(x, shape_ori)
        return x
