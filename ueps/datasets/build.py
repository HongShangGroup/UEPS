import os
from torch.utils.data import DataLoader

from .fastmrimc import FastMRIMC

def create_single_dataset(config, is_train=True):
    """create train/valid/test dataset
    """
    # set default root
    root_dict = {
        "knee":          "fastmri_knee_mc",
        "brain":         "fastmri_brain_mc",
        "stanford2d":    "stanford2d",
        "ccbrainax":     "ccbrainax",
        "ccbrainsag":    "ccbrainsag",
        "aheadaxe1":     "aheadax_e1",
        "aheadaxe2":     "aheadax_e2",
        "aheadaxe3":     "aheadax_e3",
        "aheadaxe4":     "aheadax_e4",
        "aheadaxe5":     "aheadax_e5",
        "m4rawgre":      "m4raw_gre",
    }

    which_data = config.get("which_data", "brain")
    # get root
    root = config.get("root", "../demo_data")
    root = os.path.join(root, root_dict[which_data])
    if os.path.exists(root) and os.path.isdir(root):
        print(f"data root: {root}")
    else:
        raise ValueError(f"Data root {root} does not exist.")

    # get parameters for fastmrimc
    x_type_train = config.get("x_type_train", "zfkspace")
    x_type_test = config.get("x_type_test", "zfkspace")
    y_type_train = config.get("y_type_train", "rss")
    y_type_test = config.get("y_type_test", "rss")
    take_abs_x = config.get("take_abs_x", True)
    take_abs_y = config.get("take_abs_y", True)
    mask_fixed_train = config.get("mask_fixed_train", False)
    mask_fixed_test = config.get("mask_fixed_test", False)
    mask_type = config.get("mask_type", "equispaced")
    center_frac = config.get("center_frac", 0.08)
    acc_factor = config.get("acc_factor", 4)
    offset = config.get("offset", 1)
    data_norm_type = config.get("data_norm_type", "slice_max")
    Ncoil = config.get("Ncoil", None)

    if is_train:
        train_data = FastMRIMC(
            which_data, root, "train", x_type_train, y_type_train, take_abs_x,
            take_abs_y, mask_fixed_train, mask_type, center_frac, acc_factor,
            offset, data_norm_type, Ncoil)

        valid_data = FastMRIMC(
            which_data, root, "valid", x_type_test, y_type_test, take_abs_x,
            take_abs_y, mask_fixed_test, mask_type, center_frac, acc_factor,
            offset, data_norm_type, Ncoil)

        test_data = FastMRIMC(
            which_data, root, "test", x_type_test, y_type_test, take_abs_x,
            take_abs_y, mask_fixed_test, mask_type, center_frac, acc_factor,
            offset, data_norm_type, Ncoil)
    else:
        train_data = None
        valid_data = None

        test_data = FastMRIMC(
            which_data, root, "test", x_type_test, y_type_test, take_abs_x,
            take_abs_y, mask_fixed_test, mask_type, center_frac, acc_factor,
            offset, data_norm_type, Ncoil)

    return train_data, valid_data, test_data

def build_data(config, world_size=1, is_train=True):
    """Create train/valid/test dataloaders, support DDP.

    Args:
        world_size (int): number of processes for DDP, 1 means no DDP
        is_train (bool): whether to build train/valid dataloader
    """
    train_data, valid_data, test_data = create_single_dataset(config, is_train)

    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 8)
    num_workers_valid = config.get("num_workers_valid", 6)

    if isinstance(batch_size, list):
        if len(batch_size) == 2:
            bs_train, bs_test = batch_size
        elif len(batch_size) == 1:
            bs_train, bs_test = batch_size[0], batch_size[0]
        else:
            raise RuntimeError("batch_size should have one or two values")
    elif isinstance(batch_size, int):
        bs_train, bs_test = batch_size, batch_size

    assert bs_train % world_size == 0
    assert bs_test % world_size == 0

    if is_train:
        train_loader = DataLoader(train_data,
                                    batch_size=bs_train,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=True)
        valid_loader = DataLoader(valid_data,
                                    batch_size=bs_test,
                                    shuffle=False,
                                    num_workers=num_workers_valid,
                                    pin_memory=True,
                                    drop_last=True)
    else:
        train_loader = None
        valid_loader = None

    test_loader = DataLoader(test_data, batch_size=bs_test, shuffle=False,
                             num_workers=num_workers, prefetch_factor=1, pin_memory=True)

    return train_loader, valid_loader, test_loader
