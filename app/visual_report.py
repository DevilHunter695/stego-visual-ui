from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from stego.lsb import encode_lsb, decode_lsb
from metrics.analysis import compute_metrics, dct_magnitude, wht2, chi_square_lsb
from crypto.pbkdf2_aesgcm import encrypt_aes_gcm, decrypt_aes_gcm, DEFAULT_ITERS


def visualize(
    cover_path: str,
    payload_path: Optional[str],
    out_path: str,
    fmt: str = "PNG",
    inverted: bool = False,
    keyed: bool = False,
    encrypt: bool = False,
    password: Optional[str] = None,
):
    cover_img = Image.open(cover_path).convert("RGB")
    cover_bytes = pathlib.Path(cover_path).read_bytes()

    payload_bytes: Optional[bytes] = None
    filename = None
    if payload_path:
        payload_bytes = pathlib.Path(payload_path).read_bytes()
        filename = pathlib.Path(payload_path).name

    # Optionally encrypt
    blob = payload_bytes
    salt = None
    iters = None
    encrypted_flag = False
    if payload_bytes is not None and encrypt:
        if not password:
            raise ValueError("Password required for encryption.")
        blob, salt, iters = encrypt_aes_gcm(payload_bytes, password, DEFAULT_ITERS, aad=(filename or "output.bin").encode("utf-8"))
        encrypted_flag = True

    # Encode if payload provided; otherwise assume cover already contains stego
    stego_img = None
    stego_bytes = None
    if payload_bytes is not None:
        stego_bytes = encode_lsb(
            cover_img_bytes=cover_bytes,
            filename=filename or "output.bin",
            payload=blob or b"",
            encrypted=encrypted_flag,
            inverted=inverted,
            keyed=keyed,
            out_format=fmt,
            key_material=(password.encode("utf-8") if keyed and password else None),
            salt=salt,
            iters=iters,
        )
        stego_img = Image.open(BytesIO(stego_bytes)).convert("RGB")
    else:
        stego_img = Image.open(cover_path).convert("RGB")
        stego_bytes = pathlib.Path(cover_path).read_bytes()

    # Metrics
    m = compute_metrics(cover_img, stego_img)

    # Visuals
    cover_arr = np.array(cover_img)
    stego_arr = np.array(stego_img)
    diff = np.abs(stego_arr.astype(np.int16) - cover_arr.astype(np.int16)).astype(np.uint8)
    diff_amp = (diff * 16).clip(0, 255).astype(np.uint8)

    # Histograms - LSB distribution
    cov_gray = np.mean(cover_arr, axis=2).astype(np.uint8)
    stg_gray = np.mean(stego_arr, axis=2).astype(np.uint8)
    cov_hist, _ = np.histogram(cov_gray, bins=256, range=(0, 256))
    stg_hist, _ = np.histogram(stg_gray, bins=256, range=(0, 256))

    # DCT and WHT
    dct_cov_g, dct_cov_r, dct_cov_g2, dct_cov_b = dct_magnitude(cover_img)
    dct_stg_g, dct_stg_r, dct_stg_g2, dct_stg_b = dct_magnitude(stego_img)
    wht_cov, _ = wht2(cover_img)
    wht_stg, _ = wht2(stego_img)

    # Chi-square
    chi_cov = chi_square_lsb(cover_img)
    chi_stg = chi_square_lsb(stego_img)

    # Figure
    plt.figure(figsize=(16, 12))

    # Row 1: images
    ax = plt.subplot(3, 4, 1)
    ax.imshow(cover_img)
    ax.set_title("Cover")
    ax.axis('off')

    ax = plt.subplot(3, 4, 2)
    ax.imshow(stego_img)
    ax.set_title("Stego")
    ax.axis('off')

    ax = plt.subplot(3, 4, 3)
    ax.imshow(diff_amp)
    ax.set_title("Diff ×16 (abs)")
    ax.axis('off')

    ax = plt.subplot(3, 4, 4)
    ax.axis('off')
    ax.text(0, 0.9, f"MSE: {m['MSE']:.2f}", fontsize=11)
    ax.text(0, 0.7, f"PSNR: {m['PSNR']:.2f} dB", fontsize=11)
    ax.text(0, 0.5, f"SSIM: {m['SSIM']:.4f}", fontsize=11)
    ax.text(0, 0.3, f"Keyed: {'Yes' if keyed else 'No'}", fontsize=11)
    ax.text(0, 0.1, f"Encrypted: {'Yes' if encrypt else 'No'}", fontsize=11)

    # Row 2: histograms and chi-square
    ax = plt.subplot(3, 4, 5)
    ax.plot(cov_hist, color='gray', alpha=.8, label='Cover')
    ax.plot(stg_hist, color='tab:blue', alpha=.8, label='Stego')
    ax.set_title("Grayscale Histogram")
    ax.legend(fontsize=8)

    ax = plt.subplot(3, 4, 6)
    ax.bar(['Cover', 'Stego'], [chi_cov['norm_stat'], chi_stg['norm_stat']], color=['gray','tab:blue'])
    ax.set_title("Chi-square (normalized)")

    # Row 2: DCT
    ax = plt.subplot(3, 4, 7)
    ax.imshow((dct_cov_g*255).astype(np.uint8), cmap='inferno')
    ax.set_title("DCT Gray (Cover)")
    ax.axis('off')

    ax = plt.subplot(3, 4, 8)
    ax.imshow((dct_stg_g*255).astype(np.uint8), cmap='inferno')
    ax.set_title("DCT Gray (Stego)")
    ax.axis('off')

    # Row 3: WHT
    ax = plt.subplot(3, 4, 9)
    ax.imshow((wht_cov*255).astype(np.uint8), cmap='magma')
    ax.set_title("WHT (Cover)")
    ax.axis('off')

    ax = plt.subplot(3, 4, 10)
    ax.imshow((wht_stg*255).astype(np.uint8), cmap='magma')
    ax.set_title("WHT (Stego)")
    ax.axis('off')

    # Row 3: placeholders for future plots
    ax = plt.subplot(3, 4, 11)
    ax.axis('off')
    ax = plt.subplot(3, 4, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Generate a visual stego report")
    p.add_argument("--cover", required=True, help="Cover image path (PNG/WEBP/JPG)")
    p.add_argument("--payload", help="Payload file path (optional; if omitted, analyze provided stego/cover as-is)")
    p.add_argument("--out", required=True, help="Output report image path, e.g. report.png")
    p.add_argument("--format", choices=["PNG","WEBP"], default="PNG")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--keyed", action="store_true")
    p.add_argument("--encrypt", action="store_true")
    p.add_argument("--password", help="Password (for keyed/encrypt)")
    a = p.parse_args()

    visualize(
        cover_path=a.cover,
        payload_path=a.payload,
        out_path=a.out,
        fmt=a.format,
        inverted=a.invert,
        keyed=a.keyed,
        encrypt=a.encrypt,
        password=a.password,
    )


if __name__ == "__main__":
    main()


