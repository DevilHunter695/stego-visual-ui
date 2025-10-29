# metrics/iqm.py
from __future__ import annotations
from typing import Dict

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def _to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float32)

def compute_metrics(cover: Image.Image, stego: Image.Image) -> Dict[str, float]:
    a = _to_array(cover)
    b = _to_array(stego)
    psnr = peak_signal_noise_ratio(a, b, data_range=255)
    ssim = structural_similarity(a, b, channel_axis=2, data_range=255)
    mse  = mean_squared_error(a, b)
    return {"PSNR": float(psnr), "SSIM": float(ssim), "MSE": float(mse)}