import torch
import numpy as np

def ifft2c_np(x):
    """Numpy version of ifft2c, used for k-space to image.
    fftshift and ifftshift are identical for even-length input.

    Shape:
        Input: np ndarray, shape=[Nbs, Npe, Nread], dtype=complex
        Output: np ndarray, shape=[Nbs, Npe, Nread], dtype=complex
    """
    x = np.fft.ifftshift(x, axes=(-2, -1))
    x = np.fft.ifft2(x, norm="ortho")
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x

def fft2c_np(x):
    """Numpy version of fft2c, used for image to k-space.

    Shape:
        Input: np ndarray, shape=[Nbs, Ny, Nx], dtype=complex
        Output: np ndarray, shape=[Nbs, Ny, Nx], dtype=complex
    """
    x = np.fft.fftshift(x, axes=(-2, -1))
    x = np.fft.fft2(x, norm="ortho")
    x = np.fft.ifftshift(x, axes=(-2, -1))
    return x

def ifft2c_tensor(x):
    """ifft2c with tensor as input/output, used for k-space to image.

    Shape:
        Input: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
        Output: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
    """
    x = torch.complex(x[...,0], x[...,1]).numpy()
    x = ifft2c_np(x)
    x = torch.from_numpy(x)
    x = torch.stack((x.real, x.imag), dim=-1)
    return x

def fft2c_tensor(x):
    """fft2c with tensor as input/output, used for image to k-space.

    Shape:
        Input: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
        Output: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
    """
    x = torch.complex(x[...,0], x[...,1]).numpy()
    x = fft2c_np(x)
    x = torch.from_numpy(x)
    x = torch.stack((x.real, x.imag), dim=-1)
    return x

@torch.compiler.disable(recursive=True)
def ifft2c_pt(x):
    """pytorch version of ifft2c, used for k-space to image.

    Shape:
        Input: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
        Output: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
    """
    x = torch.view_as_complex(x)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.view_as_real(x)
    return x

@torch.compiler.disable(recursive=True)
def fft2c_pt(x):
    """pytorch version of fft2c, used for image to k-space.

    Shape:
        Input: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
        Output: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
    """
    x = torch.view_as_complex(x)
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.view_as_real(x)
    return x

def prep_mask_mc(mask, Npe, Nacs):
    """
    Prepare additional mask properties for multi-coil data

    Shape:
        mask: [1, 1, Npe, 1]
        mask_idx: [Npes]
        mask_acs: [1, 1, Npe, 1]
        mask_cat: [2, 1, Npe, 1]
    """
    mask_idx = [i for i in range(Npe) if mask[:,:,i,:]>0.5]
    mask_idx = torch.from_numpy(np.array(mask_idx))

    acs_l = (Npe - Nacs + 1) // 2
    acs_r = Npe - Nacs - acs_l
    mask_acs = torch.cat((torch.zeros(1, 1, acs_l, 1),
                          torch.ones(1, 1, Nacs, 1),
                          torch.zeros(1, 1, acs_r, 1),
                          ), 2) # [1, 1, Npe, 1]

    mask_cat = torch.cat((mask, mask_acs), 0) # [2, 1, Npe, 1]
    return mask_idx, mask_acs, mask_cat

def prep_data_mc(kspace, tgt, mctgt, zfimage, mask, mask_idx, x_type, y_type, take_abs_x, take_abs_y):
    """
    Prepare multi-coil data given its type

    Shape:
        kspace: [Ncoil, Nread, Npe, 2]
        tgt: [1, Ny, Ny]
        mctgt: [Ncoil, Ny, Ny, 2]
        zfimage: [Ncoil, Ny, Ny, 2]
        mask: [1, 1, Npe, 1]
        mask_idx: [Npes]

    Details of output tensor and other parameters refer to FastMRIMC
    """
    # determine x
    if x_type == "zfkspace":
        x = kspace * mask

    elif x_type == "zfimage-mc":
        if take_abs_x:
            x = torch.abs(torch.view_as_complex(zfimage))
            x = torch.unsqueeze(x, 1)
        else:
            x = zfimage.permute(0, 3, 1, 2).contiguous()

    elif x_type == "zfimage-rss":
        x = torch.abs(torch.view_as_complex(zfimage))
        x = torch.sqrt(torch.sum(x**2, dim=0))
        x = torch.unsqueeze(x, 0)

    # determine y
    if y_type == "rss":
        y = tgt

    elif y_type == "mctgt":
        if take_abs_y:
            y = torch.abs(torch.view_as_complex(mctgt))
            y = torch.unsqueeze(y, 1)
        else:
            y = mctgt.permute(0, 3, 1, 2).contiguous()

    return x, y
