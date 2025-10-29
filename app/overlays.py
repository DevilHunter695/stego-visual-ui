from __future__ import annotations
import numpy as np
from typing import List, Tuple
from PIL import Image


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def tile_hotspots(base_img: Image.Image, diff_amp: np.ndarray, grid: int = 16, top_k: int = 8) -> Image.Image:
    """Draw red box overlays on tiles with the highest mean difference.
    diff_amp should be a HxW (or HxWx3) array of amplified absolute differences.
    """
    arr = diff_amp if diff_amp.ndim == 2 else diff_amp.mean(axis=2)
    h, w = arr.shape[:2]
    th, tw = max(1, h // grid), max(1, w // grid)
    scores: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * th, gx * tw
            y1, x1 = min(h, y0 + th), min(w, x0 + tw)
            tile = arr[y0:y1, x0:x1]
            scores.append((float(tile.mean()), (x0, y0, x1, y1)))
    scores.sort(reverse=True, key=lambda t: t[0])
    picks = scores[:top_k]
    overlay = np.array(base_img.convert("RGB"), dtype=np.uint8).copy()
    for (_, (x0, y0, x1, y1)) in picks:
        overlay[y0:y1, [x0, max(x1-1, 0)], :] = [255, 0, 0]
        overlay[[y0, max(y1-1, 0)], x0:x1, :] = [255, 0, 0]
    return to_image(overlay)


def lsb_parity_heatmap(img: Image.Image) -> Image.Image:
    """Return a grayscale heatmap of average LSB parity across RGB channels."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    lsb = (arr & 1).mean(axis=2)
    heat = (lsb * 255).astype(np.uint8)
    return to_image(heat)


def top_energy_tiles(gray_img: np.ndarray, grid: int = 16, top_k: int = 8) -> List[Tuple[int, int, int, int]]:
    """Return top-k tiles by mean value from a normalized grayscale magnitude map."""
    h, w = gray_img.shape
    th, tw = max(1, h // grid), max(1, w // grid)
    scores: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * th, gx * tw
            y1, x1 = min(h, y0 + th), min(w, x0 + tw)
            tile = gray_img[y0:y1, x0:x1]
            scores.append((float(tile.mean()), (x0, y0, x1, y1)))
    scores.sort(reverse=True, key=lambda t: t[0])
    return [bb for (_, bb) in scores[:top_k]]


def annotate_boxes(gray_img: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """Draw green boxes on a grayscale image (0..1 or 0..255); returns RGB image."""
    g = gray_img
    if g.max() <= 1.0:
        vis = (g * 255).astype(np.uint8)
    else:
        vis = g.astype(np.uint8)
    vis = np.stack([vis, vis, vis], axis=2)
    for (x0, y0, x1, y1) in boxes:
        vis[y0:y1, [x0, max(x1-1, 0)], :] = [0, 255, 0]
        vis[[y0, max(y1-1, 0)], x0:x1, :] = [0, 255, 0]
    return to_image(vis)


