import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def mae(gt: np.ndarray, pred: np.ndarray):
    """Compute Mean Absolute Error (MAE)"""
    return np.mean(np.abs(gt - pred))

def mse(gt: np.ndarray, pred: np.ndarray):
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)

def nmse(gt: np.ndarray, pred: np.ndarray):
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt: np.ndarray, pred: np.ndarray, maxval: float):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(gt: np.ndarray, pred: np.ndarray, maxval: float):
    """Compute Structural Similarity Index Metric (SSIM)"""
    assert gt.ndim == 3
    Nc, Ny, Nx = gt.shape
    y = 0.0
    for i in range(Nc):
        y = y + structural_similarity(gt[i,:,:], pred[i,:,:], data_range=maxval)
    y = y / Nc
    return y
