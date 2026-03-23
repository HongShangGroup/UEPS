import h5py
import torch
from pathlib import Path
from typing import Union
from torchvision.datasets.vision import VisionDataset
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type as create_mask

from .common import ifft2c_tensor, fft2c_tensor
from .common import prep_mask_mc, prep_data_mc

class FastMRIMC(VisionDataset):
    """
    A Dataset based on fastMRI multi-coil data, treat each slice as a 2D data.

    Main differences from fastmri.data.SliceDataset:
    1) Output (x, y, mask), x is model input, with options as below:
        - "zfkspace": undersampled kspace with zero-filling, [Ncoil, Nread, Npe, 2]
        - "zfimage-mc": ifft of zero-filled kspace, [Ncoil, 1 or 2, Nx, Nx]
        - "zfimage-rss": root-sum-of-squares combination of above, [1, Nx, Nx]

        y is target recon image, with options as below:
        - "mctgt": multi-coil target by ifft of full-sampled kspace, [Ncoil, 1 or 2, Ny, Ny]
        - "rss": root-sum-of-squares combination of above, [1, Ny, Ny]

        mask is concat of kspace undersampling mask and ACS mask [2, 1, Npe, 1]

        dimension definitions:
        - Ncoil: number of coils
        - Npe: number of phase encoding
        - Npes: number of phase encoding after sampling
        - Nacs: number of phase encoding for acs (autocalibration)
        - Nread: number of readout samples per phase encoding
        - Ny: target image size (Ny, Ny), typically Ny = 320
        - Nx: input image size (Nx, Nx), typically Nx = Ny

    2) Fixed shape across all data for both kspace and image
        Ncoil, Nread, Npe, Nx, Ny are consistent across all data.
        Additionally, Nread = Npe = Nx = Ny

    3) Allow batch size > 1

    Args:
        which_data (str): choose from "knee", "brain", and more options as below
        root (str or Path): directory where data exists
        split (str): one of {"train", "valid", "test"} for data split
        x_type (str): one of {"zfkspace", "zfimage-mc", "zfimage-rss"}
        y_type (str): one of {"rss", "mctgt"}
        take_abs_x (bool): if True, image x is magnitude, otherwise x is complex
        take_abs_y (bool): if True, image y is magnitude, otherwise y is complex
        mask_fixed (bool): if True, fix mask across all data
        mask_type (str): options refer to fastmri.data.subsample
        center_frac (float): fraction of k-space center to include
        acc_factor (int): acceleration factor
        offset (int): offset from 0 to begin masking
        data_norm_type (str): one of {"none", "volume_max", "slice_max", "avg"}
        Ncoil (int): maximum Ncoil, if None, use default value
    """

    split_map = {
        "brain": {"train": "multicoil_train", "valid": "multicoil_val",  "test": "multicoil_test_full"},
        "knee":  {"test": "test"},
        "stanford2d":    {"test": "test"},
        "ccbrainax":     {"test": "test"},
        "ccbrainsag":    {"test": "test"},
        "aheadaxe1":     {"test": "test"},
        "aheadaxe2":     {"test": "test"},
        "aheadaxe3":     {"test": "test"},
        "aheadaxe4":     {"test": "test"},
        "aheadaxe5":     {"test": "test"},
        "m4rawgre":      {"test": "test"},
    }
    Ncoil = {
        "knee": 15,
        "brain": 24,
        "stanford2d": 32,
        "ccbrainax": 12,
        "ccbrainsag": 12,
        "aheadaxe1": 32,
        "aheadaxe2": 32,
        "aheadaxe3": 32,
        "aheadaxe4": 32,
        "aheadaxe5": 32,
        "m4rawgre": 4,
    }
    avg_max_value = {
        "knee": 0.0004,
        "brain": 0.000835,
        "stanford2d": 1348138.72,
        "ccbrainax": 30488595.85,
        "ccbrainsag": 30620699.87,
        "aheadaxe1": 13859689984.0,
        "aheadaxe2": 25906087987.2,
        "aheadaxe3": 16932815411.2,
        "aheadaxe4": 12756087961.6,
        "aheadaxe5": 11186915737.6,
        "m4rawgre": 313.52,
    }
    Ny = 320
    Npe = 320
    Nread = 320
    seed = 16

    def __init__(self,
                 which_data: str = "brain",
                 root: Union[str, Path] = "../demo_data/fastmri_brain_mc",
                 split: str = "train",
                 x_type: str = "zfkspace",
                 y_type: str = "rss",
                 take_abs_x: bool = False,
                 take_abs_y: bool = True,
                 mask_fixed: bool = True,
                 mask_type: str = "equispaced",
                 center_frac: float = 0.08,
                 acc_factor: int = 4,
                 offset: int = None,
                 data_norm_type: str = "avg",
                 Ncoil: int = None,
                 ):
        super().__init__()
        assert which_data in list(self.split_map.keys())
        assert split in ["train", "valid", "test"]
        data_path = Path(root) / self.split_map[which_data][split]

        self.slicedata = FastMRISliceH5PY(data_path, "multicoil", False)

        print(f"FastMRI-{which_data} {split} has {len(self.slicedata.raw_samples)} data")

        assert x_type in ["zfkspace", "zfimage-mc", "zfimage-rss"]
        assert y_type in ["rss", "mctgt"]
        assert data_norm_type in ["none", "volume_max", "slice_max", "avg"]

        self.x_type = x_type
        self.y_type = y_type
        self.take_abs_x = take_abs_x
        self.take_abs_y = take_abs_y
        self.mask_fixed = mask_fixed
        self.which_data = which_data
        self.data_norm_type = data_norm_type
        self.crop_shape = (self.Ny, self.Ny)
        self.offset = offset

        assert isinstance(center_frac, float) or isinstance(center_frac, list)
        assert isinstance(acc_factor, int) or isinstance(acc_factor, list)
        center_frac = [center_frac] if isinstance(center_frac, float) else center_frac
        acc_factor = [acc_factor] if isinstance(acc_factor, int) else acc_factor

        self.mask_func = create_mask(mask_type, center_frac, acc_factor)

        shape = (1, self.Nread, self.Npe, 2)
        self.mask, self.Nacs = self.mask_func(shape, self.offset, self.seed)
        self.Ncoil_max = Ncoil if isinstance(Ncoil, int) else self.Ncoil[which_data]

    def _crop_if_needed(self, image):
        # expect input shape [Ncoil, Nread, Npe, 2]
        if self.crop_shape[0] < image.shape[-3]:
            h_from = (image.shape[-3] - self.crop_shape[0]) // 2
            h_to = h_from + self.crop_shape[0]
        else:
            h_from = 0
            h_to = image.shape[-3]

        if self.crop_shape[1] < image.shape[-2]:
            w_from = (image.shape[-2] - self.crop_shape[1]) // 2
            w_to = w_from + self.crop_shape[1]
        else:
            w_from = 0
            w_to = image.shape[-2]

        return image[:, h_from:h_to, w_from:w_to, :]

    def _pad_if_needed(self, image):
        # expect input shape [Ncoil, Nread, Npe, 2]
        pad_h = self.crop_shape[0] - image.shape[-3]
        pad_w = self.crop_shape[1] - image.shape[-2]

        if pad_w > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            pad_left = pad_right = 0

        if pad_h > 0:
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
        else:
            pad_up = pad_down = 0

        pad = (pad_left, pad_right, pad_up, pad_down)
        image = image.permute(0, 3, 1, 2)
        image = torch.nn.functional.pad(image, pad, mode='reflect')
        image = image.permute(0, 2, 3, 1).contiguous()
        return image

    def _to_uniform_size(self, image):
        # expect input shape [Ncoil, Nread, Npe, 2]
        image = self._crop_if_needed(image)
        shape_raw = torch.tensor([image.shape[-3], image.shape[-2]])
        image = self._pad_if_needed(image)
        return image, shape_raw

    def __len__(self) -> int:
        return len(self.slicedata.raw_samples)

    def __getitem__(self, index: int):
        # get original kspace
        kspace_raw, max_value = self.slicedata.__getitem__(index)
        kspace_raw = torch.from_numpy(kspace_raw) # [Ncoil, Nread, Npe] complex
        kspace_raw = torch.view_as_real(kspace_raw) # [Ncoil, Nread, Npe, 2]
        Ncoil, Nread, Npe, _ = kspace_raw.shape

        # fix Ncoil, zero pad if not enough
        if Ncoil < self.Ncoil_max:
            kspace_zp = torch.zeros((self.Ncoil_max-Ncoil), Nread, Npe, 2)
            kspace = torch.cat((kspace_raw, kspace_zp), 0)

        elif Ncoil == self.Ncoil_max:
            kspace = kspace_raw

        elif Ncoil > self.Ncoil_max:
            kspace = torch.index_select(kspace_raw, 0, torch.randperm(Ncoil)[0:self.Ncoil_max])

        # fix shape of kspace & target
        mctgt = ifft2c_tensor(kspace) # [Ncoil, Nread, Npe, 2]
        mctgt, shape_raw = self._to_uniform_size(mctgt) # [Ncoil, Ny, Ny, 2]

        # data normalization
        if self.data_norm_type == "volume_max":
            norm_factor = 1.0 / max_value
        elif self.data_norm_type == "slice_max":
            norm_factor = 1.0 / torch.max(torch.sqrt(torch.sum(mctgt**2, dim=(0, -1)))).item()
        elif self.data_norm_type == "avg":
            norm_factor = 1.0 / self.avg_max_value[self.which_data]
        elif self.data_norm_type == "none":
            norm_factor = 1.0

        mctgt = mctgt * norm_factor
        max_value = max_value * norm_factor
        max_value = torch.tensor(max_value, dtype=torch.float32)

        # get kspace and rss
        kspace = fft2c_tensor(mctgt)
        tgt = torch.abs(torch.view_as_complex(mctgt))
        tgt = torch.sqrt(torch.sum(tgt**2, dim=0))
        tgt = torch.unsqueeze(tgt, 0)

        # get undersampling mask
        if self.mask_fixed:
            mask, Nacs = self.mask, self.Nacs
        else:
            mask, Nacs = self.mask_func((1, self.Nread, self.Npe, 2), self.offset, None)

        mask_idx, mask_acs, mask_cat = prep_mask_mc(mask, self.Npe, Nacs)

        # get multi-coil zero-filled image
        if "zfimage" in self.x_type:
            zfimage = ifft2c_tensor(kspace * mask) # [Ncoil, Nread_new, Npe_new, 2]
        else:
            zfimage = None

        # determine x,y given its type
        x, y = prep_data_mc(kspace, tgt, mctgt, zfimage, mask, mask_idx, self.x_type,
                            self.y_type, self.take_abs_x, self.take_abs_y)

        return x, y, mask_cat, max_value, shape_raw


class FastMRISliceH5PY(SliceDataset):
    def __init__(self,
                 root: Union[str, Path],
                 challenge: str = "multicoil",
                 use_dataset_cache: bool = False,
                 ):

        super().__init__(root=Path(root), challenge=challenge, use_dataset_cache=use_dataset_cache)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, index: int):
        fname, dataslice, metadata = self.raw_samples[index]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            max_value = attrs["max"]
        return kspace, max_value
