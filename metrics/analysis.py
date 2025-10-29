import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dctn
from scipy.linalg import hadamard

# ------------------------------------------------------------
# Quality Metrics (PSNR, MSE, SSIM)
# ------------------------------------------------------------
def compute_metrics(cover_img, stego_img):
    a = np.array(cover_img, dtype=np.float32)
    b = np.array(stego_img, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else 100
    ssim_val = ssim(a, b, channel_axis=2, data_range=255)
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim_val}

# ------------------------------------------------------------
# DCT Magnitude (log-scaled)
# ------------------------------------------------------------
def dct_magnitude(im):
    arr = np.array(im.convert("RGB"), dtype=np.float32)
    gray = np.mean(arr, axis=2)
    dct_gray = np.log(np.abs(dctn(gray, norm="ortho")) + 1)
    dct_r = np.log(np.abs(dctn(arr[:,:,0], norm="ortho")) + 1)
    dct_g = np.log(np.abs(dctn(arr[:,:,1], norm="ortho")) + 1)
    dct_b = np.log(np.abs(dctn(arr[:,:,2], norm="ortho")) + 1)
    return (
        dct_gray / dct_gray.max(),
        dct_r / dct_r.max(),
        dct_g / dct_g.max(),
        dct_b / dct_b.max()
    )

# ------------------------------------------------------------
# Walsh–Hadamard Transform (WHT)
# ------------------------------------------------------------
def wht2(im):
    arr = np.array(im.convert("L"), dtype=np.float32)
    n = 1 << int(np.ceil(np.log2(max(arr.shape))))
    padded = np.zeros((n, n))
    padded[:arr.shape[0], :arr.shape[1]] = arr
    H = hadamard(n)
    WHT = H @ padded @ H
    mag = np.log(np.abs(WHT) + 1)
    return mag / mag.max(), (n, n)

# ------------------------------------------------------------
# Chi-square LSB analysis
# ------------------------------------------------------------
def chi_square_lsb(img: Image.Image):
    arr = np.array(img.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    even = hist[::2]
    odd = hist[1::2]
    expected = (even + odd) / 2
    chi2_stat = np.sum(((even - expected)**2) / (expected + 1e-6))
    norm_stat = chi2_stat / len(even)
    return {"chi2_stat": float(chi2_stat), "norm_stat": float(norm_stat)}