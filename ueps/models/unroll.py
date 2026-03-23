import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor

from .img2img import Img2Img
from .ft import IFFT2m, FFT2m, complex_mul


class Unroll(nn.Module):
    """Unrolling with smaps estimation to reduce coil dimension.

    Args:
        Ncascade (int): number of cascade
        Nread (int): number of readout samples
        Npe (int): number of phase encoding
        dc_type (str): type of data consistency
        img_model_config (dict): dict of parameters for img2img model
        sen_model_config (dict): dict of parameters for model of learning samps

    Shape: N, M, R, P stand for batch, multi-coil, readout, phase-encoding
        Input:
            kspace: [N, M, R, P, 2]
            mask_all: [N, 1, 1, P, 1]
            mask_acs: [N, 1, 1, P, 1]
        Output:
            recon: [N, 1, R, P]
    """
    def __init__(self,
                 Ncascade: int = 4,
                 Nread: int = 320,
                 Npe: int = 320,
                 dc_type: str = "soft",
                 img_model_config: dict = {},
                 sen_model_config: dict = {},
                 ):
        super().__init__()
        self.Ncascade = Ncascade
        self.ift = IFFT2m()

        config_list = []
        for i in range(Ncascade):
            name_i = f"img_model_config_{i}"
            config_i = name_i if name_i in img_model_config else "img_model_config"
            config_list.append(img_model_config[config_i])

        self.get_smaps = LearnSenMap(sen_model_config)
        self.blocks = nn.ModuleList([
            UIBlock(config_list[i], Nread, Npe, dc_type) for i in range(Ncascade)
        ])

    def forward(self, kspace: Tensor, mask_all: Tensor, mask_acs: Tensor) -> Tensor:
        smaps = self.get_smaps(kspace, mask_acs)
        ref_kspace = kspace.clone()
        x = self.ift(kspace)
        for block in self.blocks:
            x = block(x, ref_kspace, mask_all, smaps) # [N, M, R, P, 2]
        x = torch.sqrt(torch.sum(x**2, dim=(1, -1))).unsqueeze(1) # [N, 1, R, P]
        return x


class UnrollE(nn.Module):
    """Unrolling without smaps estimation or coil dimension reduction.

    Args:
        Ncascade (int): number of cascade
        Nread (int): number of readout samples
        Npe (int): number of phase encoding
        dc_type (str): type of data consistency
        use_coil_mask (bool): if True, ensure zero-padded coils remain zero
        img_model_config (dict): dict of parameters for img2img model

    Shape: N, M, R, P stand for batch, multi-coil, readout, phase-encoding
        Input:
            kspace: [N, M, R, P, 2]
            mask_all: [N, 1, 1, P, 1]
        Output:
            recon: [N, 1, R, P]
    """
    def __init__(self,
                 Ncascade: int = 4,
                 Nread: int = 320,
                 Npe: int = 320,
                 dc_type: str = "soft",
                 use_coil_mask: bool = True,
                 img_model_config: dict = {},
                 ):
        super().__init__()
        self.Ncascade = Ncascade
        self.use_coil_mask = use_coil_mask
        self.ift = IFFT2m()

        config_list = []
        for i in range(Ncascade):
            name_i = f"img_model_config_{i}"
            config_i = name_i if name_i in img_model_config else "img_model_config"
            config_list.append(img_model_config[config_i])

        self.blocks = nn.ModuleList([
            UEBlock(config_list[i], Nread, Npe, dc_type) for i in range(Ncascade)
        ])

    def forward(self, kspace: Tensor, mask_all: Tensor) -> Tensor:
        target_dtype = kspace.dtype

        ref_kspace = kspace.clone()
        with torch.autocast(device_type='cuda', enabled=False):
            x = self.ift(kspace.float())
        x = x.to(dtype=target_dtype)

        # generate a mask for zero-padded coil
        if self.use_coil_mask:
            with torch.no_grad():
                coil_mask = torch.sum(x**2, dim=(-3, -2, -1), keepdim=True)
                coil_mask = (coil_mask != 0).float() # [N, M, 1, 1, 1]

        for block in self.blocks:
            x = block(x, ref_kspace, mask_all) # [N, M, R, P, 2]

        if self.use_coil_mask:
            x = x * coil_mask

        x = torch.sqrt(torch.sum(x**2, dim=(1, -1))).unsqueeze(1) # [N, 1, R, P]
        return x


class UnrollEP(nn.Module):
    """Unrolling Expanded model with progressive resolution adjustment.

    Args:
        Ncascade (int): number of cascade
        Nread (int): number of readout samples
        Npe (int): number of phase encoding
        dc_type (str): type of data consistency
        use_coil_mask (bool): if True, ensure zero-padded coils remain zero
        resolution_schedule (list of tuple): list of target resolutions for each cascade, including the initial one. If None, no resolution adjustment.
        img_model_config (dict): dict of parameters for img2img model

    Shape: N, M, R, P stand for batch, multi-coil, readout, phase-encoding
        Input:
            kspace: [N, M, R, P, 2]
            mask_all: [N, 1, 1, P, 1]
        Output:
            recon: [N, 1, R, P]
    """
    def __init__(self,
                 Ncascade: int = 4,
                 Nread: int = 320,
                 Npe: int = 320,
                 dc_type: str = "soft",
                 use_coil_mask: bool = True,
                 resolution_schedule=None,
                 img_model_config: dict = {},
                 ):
        super().__init__()
        self.Ncascade = Ncascade
        self.use_coil_mask = use_coil_mask
        self.ift = IFFT2m()
        self.ft = FFT2m()
        self.resolution_schedule = resolution_schedule
        self.start_res = (Nread, Npe) if resolution_schedule is None else resolution_schedule[0]

        config_list = []
        for i in range(Ncascade):
            name_i = f"img_model_config_{i}"
            config_i = name_i if name_i in img_model_config else "img_model_config"
            config_list.append(img_model_config[config_i])

        self.blocks = nn.ModuleList([
            UEPBlock(
                config_list[i], 
                target_res=resolution_schedule[i+1] if resolution_schedule else (Nread, Npe), 
                dc_type=dc_type,
            )
            for i in range(Ncascade)
        ])

    def _crop_if_needed(self, image):
        # expect input shape [N, M, R, P, 2]
        if self.start_res[0] < image.shape[-3]:
            h_from = (image.shape[-3] - self.start_res[0]) // 2
            h_to = h_from + self.start_res[0]
        else:
            h_from = 0
            h_to = image.shape[-3]

        if self.start_res[1] < image.shape[-2]:
            w_from = (image.shape[-2] - self.start_res[1]) // 2
            w_to = w_from + self.start_res[1]
        else:
            w_from = 0
            w_to = image.shape[-2]

        return image[:, :, h_from:h_to, w_from:w_to, :]

    def forward(self, kspace: Tensor, mask_all: Tensor) -> Tensor:
        target_dtype = kspace.dtype

        # kspace: [N, M, R, P, 2], mask_all: [N, 1, 1, P, 1]
        ref_kspace_full = kspace.clone()
        mask_full = mask_all.clone()
        with torch.autocast(device_type='cuda', enabled=False):
            x = self.ift(kspace.float()) # [N, M, R, P, 2]
        x = x.to(dtype=target_dtype)

        # generate a mask for zero-padded coil
        if self.use_coil_mask:
            with torch.no_grad():
                coil_mask = torch.sum(x**2, dim=(-3, -2, -1), keepdim=True)
                coil_mask = (coil_mask != 0).float() # [N, M, 1, 1, 1]

        # adjust initial resolution if needed
        kspace = self._crop_if_needed(kspace)

        with torch.autocast(device_type='cuda', enabled=False):
            x = self.ift(kspace.float())
        x = x.to(dtype=target_dtype)

        for i, block in enumerate(self.blocks):
            x = block(x, ref_kspace_full, mask_full)

        if self.use_coil_mask:
            x = x * coil_mask

        x = torch.sqrt(torch.sum(x**2, dim=(1, -1))).unsqueeze(1) # [N, 1, R, P]
        return x
    

class UEPBlock(nn.Module):
    """one step for unrollEP
    """
    def __init__(self,
                 model_config: dict,
                 target_res: tuple = (320, 320),
                 dc_type: str = "soft",
                 ):
        super().__init__()
        self.net = Img2Img(model_config)
        self.ift = IFFT2m()
        self.ft = FFT2m()
        self.Nread, self.Npe = target_res

        assert dc_type in ["soft", "hard", "learn0d", "learn1d", "learn2d",
                           "sigmoid0d", "sigmoid1d", "sigmoid2d"]
        self.dc_type = dc_type

        if self.dc_type in ["soft", "learn0d", "sigmoid0d"]:
            self.dc_weight = nn.Parameter(torch.ones(1))

        elif self.dc_type in ["learn1d", "sigmoid1d"]:
            self.dc_weight = nn.Parameter(torch.ones(1,1,1,self.Nread,1))

        elif self.dc_type in ["learn2d", "sigmoid2d"]:
            self.dc_weight = nn.Parameter(torch.ones(1,1,self.Nread,self.Npe,1))

    def data_consistency(self, kspace: Tensor, ref_kspace: Tensor, mask: Tensor) -> Tensor:
        if "soft" in self.dc_type:
            zero = torch.zeros(1, 1, 1, 1, 1).to(kspace)
            soft_dc = torch.where(mask.to(torch.bool), kspace-ref_kspace, zero) * self.dc_weight
            kspace = kspace - soft_dc

        elif "hard" in self.dc_type:
            kspace = kspace * (1-mask) + ref_kspace * mask

        elif "learn" in self.dc_type:
            kspace = kspace - (kspace - ref_kspace) * mask * self.dc_weight

        elif "sigmoid" in self.dc_type:
            kspace = kspace - (kspace - ref_kspace) * mask * F.sigmoid(self.dc_weight)

        return kspace

    def forward(self, x: Tensor, ref_kspace_full: Tensor, mask_full: Tensor):
        """
        Args:
            x: [N, M, H, W, 2] image at current resolution
            ref_kspace_full: [N, M, H_full, W_full, 2] full resolution reference kspace
            mask_full: [N, 1, 1, W_full, 1] full resolution mask
        """
        target_dtype = x.dtype

        # 1. net processes at current resolution
        x = self.net(x)

        with torch.autocast(device_type='cuda', enabled=False):
            kspace = self.ft(x.float())
        kspace = kspace.to(dtype=target_dtype)

        # 2. crop ref_kspace/mask to current resolution (center crop)
        N, M, H, W, _ = kspace.shape
        Hf, Wf = ref_kspace_full.shape[-3], ref_kspace_full.shape[-2]
        h_from = (Hf - H) // 2
        w_from = (Wf - W) // 2
        ref_kspace_crop = ref_kspace_full[:, :, h_from:h_from+H, w_from:w_from+W, :]
        mask_crop = mask_full[:, :, :, w_from:w_from+W, :]

        # 3. DC at current resolution (with ref_kspace)
        kspace_dc = self.data_consistency(kspace, ref_kspace_crop, mask_crop)

        # 4. pad kspace to target resolution, fill padded region with ref_kspace_full
        if self.Nread > H and self.Npe > W:
            Ht, Wt = self.Nread, self.Npe
            pad_h = Ht - H
            pad_w = Wt - W
            pad_left = pad_w // 2 if pad_w > 0 else 0
            pad_right = pad_w - pad_left if pad_w > 0 else 0
            pad_up = pad_h // 2 if pad_h > 0 else 0
            pad_down = pad_h - pad_up if pad_h > 0 else 0

            # pad kspace_dc with zeros first
            kspace_padded = kspace_dc.permute(0, 1, 4, 2, 3)  # [N, M, 2, H, W]
            kspace_padded = F.pad(kspace_padded, (pad_left, pad_right, pad_up, pad_down), mode='constant', value=0)
            kspace_padded = kspace_padded.permute(0, 1, 3, 4, 2).contiguous()  # [N, M, Ht, Wt, 2]

            # get ref_kspace at target resolution
            ht_from = (Hf - Ht) // 2
            wt_from = (Wf - Wt) // 2
            ref_kspace_target = ref_kspace_full[:, :, ht_from:ht_from+Ht, wt_from:wt_from+Wt, :]

            # fill padded regions with ref_kspace values
            if pad_up > 0:
                kspace_padded[:, :, :pad_up, :, :] = ref_kspace_target[:, :, :pad_up, :, :]
            if pad_down > 0:
                kspace_padded[:, :, -pad_down:, :, :] = ref_kspace_target[:, :, -pad_down:, :, :]
            if pad_left > 0:
                kspace_padded[:, :, :, :pad_left, :] = ref_kspace_target[:, :, :, :pad_left, :]
            if pad_right > 0:
                kspace_padded[:, :, :, -pad_right:, :] = ref_kspace_target[:, :, :, -pad_right:, :]
        else:
            kspace_padded = kspace_dc

        # 5. ift to get image at next resolution
        with torch.autocast(device_type='cuda', enabled=False):
            x_next = self.ift(kspace_padded.float())
        x_next = x_next.to(dtype=target_dtype)

        return x_next
    

class UEBlock(nn.Module):
    """one step for unrollE
    """
    def __init__(self,
                 model_config: dict,
                 Nread: int = 320,
                 Npe: int = 320,
                 dc_type: str = "soft",
                 ):
        super().__init__()
        self.net = Img2Img(model_config)
        self.ift = IFFT2m()
        self.ft = FFT2m()

        assert dc_type in ["soft", "hard", "learn0d", "learn1d", "learn2d",
                           "sigmoid0d", "sigmoid1d", "sigmoid2d"]
        self.dc_type = dc_type

        if self.dc_type in ["soft", "learn0d", "sigmoid0d"]:
            self.dc_weight = nn.Parameter(torch.ones(1))

        elif self.dc_type in ["learn1d", "sigmoid1d"]:
            self.dc_weight = nn.Parameter(torch.ones(1,1,1,Npe,1))

        elif self.dc_type in ["learn2d", "sigmoid2d"]:
            self.dc_weight = nn.Parameter(torch.ones(1,1,Nread,Npe,1))

    def data_consistency(self, kspace: Tensor, ref_kspace: Tensor, mask: Tensor) -> Tensor:
        if "soft" in self.dc_type:
            zero = torch.zeros(1, 1, 1, 1, 1).to(kspace)
            soft_dc = torch.where(mask.to(torch.bool), kspace-ref_kspace, zero) * self.dc_weight
            kspace = kspace - soft_dc

        elif "hard" in self.dc_type:
            kspace = kspace * (1-mask) + ref_kspace * mask

        elif "learn" in self.dc_type:
            kspace = kspace - (kspace - ref_kspace) * mask * self.dc_weight

        elif "sigmoid" in self.dc_type:
            kspace = kspace - (kspace - ref_kspace) * mask * F.sigmoid(self.dc_weight)

        return kspace

    def forward(self, x: Tensor, ref_kspace: Tensor, mask: Tensor) -> Tensor:
        target_dtype = x.dtype

        x = self.net(x)

        with torch.autocast(device_type='cuda', enabled=False):
            kspace = self.ft(x.float())
        kspace = kspace.to(dtype=target_dtype)

        kspace = self.data_consistency(kspace, ref_kspace, mask)

        with torch.autocast(device_type='cuda', enabled=False):
            x = self.ift(kspace.float())
        x = x.to(dtype=target_dtype)
        return x


class UIBlock(UEBlock):
    """one step for unroll with image as intermediate
    """
    def image_model(self, x: Tensor, smaps: Tensor) -> Tensor:
        x1 = reduce_smaps(x, smaps) # [N, R, P, 2]
        x1 = self.net(x1) # [N, R, P, 2]
        x1 = expand_smaps(x1, smaps) # [N, M, R, P, 2]
        x = x + x1
        return x

    def forward(self, x: Tensor, ref_kspace: Tensor, mask: Tensor, smaps: Tensor) -> Tensor:
        x = self.image_model(x, smaps)
        kspace = self.ft(x)
        kspace = self.data_consistency(kspace, ref_kspace, mask)
        x = self.ift(kspace)
        return x


def reduce_smaps(x, smaps):
    """Reduce multi-coil images into one image with sensitivity maps.

    Shape: N, M, R, P stand for batch, multi-coil, readout, phase-encoding
        Input:
            x: [N, M, R, P, 2]
            smaps: [N, M, R, P, 2]
        Output:
            x: [N, R, P, 2]
    """
    smaps = torch.stack((smaps[...,0], -smaps[...,1]), dim=-1)
    x = complex_mul(x, smaps)
    x = torch.sum(x, dim=1)
    return x


def expand_smaps(x, smaps):
    """Expand one image into multi-coil images with sensitivity maps.

    Shape: N, M, R, P stand for batch, multi-coil, readout, phase-encoding
        Input:
            x: [N, R, P, 2]
            smaps: [N, M, R, P, 2]
        Output:
            x: [N, M, R, P, 2]
    """
    x = torch.unsqueeze(x, 1)
    x = complex_mul(x, smaps)
    return x


class LearnSenMap(nn.Module):
    """Learning sensitivity map from multi-coil k-space center

    Shape: N, M, R, P stand for batch, multi-coil, readout, phase-encoding
        Input:
            kspace: [N, M, R, P, 2]
            mask: [N, 1, 1, P, 1]
        Output:
            smaps: [N, M, R, P, 2]
    """
    def __init__(self, model_config: dict):
        super().__init__()
        self.net = Img2Img(model_config)
        self.ift = IFFT2m()

        self.use_coil_mask = model_config.get("use_coil_mask", True)
        self.smaps_const_init = model_config.get("smaps_const_init", True)
        const_init = torch.stack((torch.ones(1,1,1,1), torch.zeros(1,1,1,1)), dim=-1)
        self.register_buffer('const_init', const_init, persistent=False)

    def forward(self, kspace: Tensor, mask: Tensor) -> Tensor:
        x = self.ift(kspace * mask)

        if self.use_coil_mask:
            with torch.no_grad():
                coil_mask = torch.sum(x**2, dim=(-3, -2, -1), keepdim=True)
                coil_mask = (coil_mask != 0).float() # [N, M, 1, 1, 1]

        x = self.const_init + self.net(x) if self.smaps_const_init else self.net(x)
        x = x * coil_mask if self.use_coil_mask else x
        rss = torch.sqrt(torch.sum(x**2, dim=(1, -1), keepdim=True))
        x = x / rss
        return x
