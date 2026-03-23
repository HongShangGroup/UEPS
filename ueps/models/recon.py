import torch
import torch.nn as nn
Tensor = torch.Tensor

from .unroll import Unroll, UnrollE, UnrollEP


class ReconFramework(nn.Module):
    """choose recon framework and set default parameters"""

    def __init__(self, config):
        super().__init__()
        recon_type = config.get("recon_type", "unroll").lower()
        assert recon_type in ["unroll", "unrolle", "unrollep"]
        self.recon_type = recon_type

        if self.recon_type == "unroll":
            self.Recon = Unroll(
                Ncascade=config.get("Ncascade", 8),
                Nread=config.get("Ny", 320),
                Npe=config.get("Nx", 320),
                dc_type=config.get("dc_type", "soft"),
                img_model_config=config,
                sen_model_config=config.get("sen_model_config"),
            )

        elif self.recon_type == "unrolle":
            self.Recon = UnrollE(
                Ncascade=config.get("Ncascade", 8),
                Nread=config.get("Ny", 320),
                Npe=config.get("Nx", 320),
                dc_type=config.get("dc_type", "soft"),
                use_coil_mask=config.get("use_coil_mask", True),
                img_model_config=config,
            )

        elif self.recon_type == "unrollep":
            self.Recon = UnrollEP(
                Ncascade=config.get("Ncascade", 8),
                Nread=config.get("Ny", 320),
                Npe=config.get("Nx", 320),
                dc_type=config.get("dc_type", "soft"),
                use_coil_mask=config.get("use_coil_mask", True),
                resolution_schedule=config.get("resolution_schedule", None),
                img_model_config=config,
            )

        mask_all_idx = torch.tensor([0])
        mask_acs_idx = torch.tensor([1])
        self.register_buffer('mask_all_idx', mask_all_idx, persistent=False)
        self.register_buffer('mask_acs_idx', mask_acs_idx, persistent=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        mask_all = torch.index_select(mask, 1, self.mask_all_idx)
        mask_acs = torch.index_select(mask, 1, self.mask_acs_idx)

        if self.recon_type == "unroll":
            x = self.Recon(x, mask_all, mask_acs)

        elif self.recon_type in ["unrolle", "unrollep"]:
            x = self.Recon(x, mask_all)

        return x
