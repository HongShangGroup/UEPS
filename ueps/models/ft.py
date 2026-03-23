import torch
import torch.nn as nn
Tensor = torch.Tensor

from ..datasets.common import ifft2c_pt, fft2c_pt

class FFT2m(nn.Module):
    """2D FT with additional multi-coil dim treated as batch dim

    Shape:
        Input: tensor, shape=[Nbs, Ncoil, Ny, Nx, 2], dtype=float
        Output: tensor, shape=[Nbs, Ncoil, Ny, Nx, 2], dtype=float
    """
    def __init__(self):
        super().__init__()
        self.ft2 = FFT2()

    def forward(self, x: Tensor) -> Tensor:
        N, M, H, W, _ = x.shape
        x = x.view(-1, H, W, 2)
        x = self.ft2(x)
        x = x.view(N, M, H, W, 2)
        return x


class IFFT2m(nn.Module):
    """Inverse 2D FT with additional multi-coil dim treated as batch dim

    Shape:
        Input: tensor, shape=[Nbs, Ncoil, Npe, Nread, 2], dtype=float
        Output: tensor, shape=[Nbs, Ncoil, Npe, Nread, 2], dtype=float
    """
    def __init__(self):
        super().__init__()
        self.ift2 = IFFT2()

    def forward(self, x: Tensor) -> Tensor:
        N, M, H, W, _ = x.shape
        x = x.view(-1, H, W, 2)
        x = self.ift2(x)
        x = x.view(N, M, H, W, 2)
        return x


class FFT2(nn.Module):
    """2D FFT layer

    Shape:
        Input: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
        Output: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
    """
    def forward(self, x: Tensor) -> Tensor:
        x = fft2c_pt(x)
        return x


class IFFT2(nn.Module):
    """2D inverse FFT layer

    Shape:
        Input: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
        Output: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
    """
    def forward(self, x: Tensor) -> Tensor:
        x = ifft2c_pt(x)
        return x


def complex_matmul(A, B):
    A_real, A_imag = A[..., 0], A[..., 1]
    B_real, B_imag = B[..., 0], B[..., 1]
    real_part = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
    imag_part = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)
    result = torch.stack((real_part, imag_part), dim=-1)
    return result


def complex_mul(A, B):
    A_real, A_imag = A[..., 0], A[..., 1]
    B_real, B_imag = B[..., 0], B[..., 1]
    real_part = A_real * B_real - A_imag * B_imag
    imag_part = A_real * B_imag + A_imag * B_real
    result = torch.stack((real_part, imag_part), dim=-1)
    return result
