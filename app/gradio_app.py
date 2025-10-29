from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import gradio as gr

from app.api import hide_file_in_image, extract_file_from_image
from metrics.analysis import compute_metrics, dct_magnitude, wht2, chi_square_lsb
from app.overlays import to_image, tile_hotspots, lsb_parity_heatmap, top_energy_tiles, annotate_boxes
import matplotlib.pyplot as plt
from stego.lsb import capacity_bytes


def diff_amp_image(cover_img: Image.Image, stego_img: Image.Image, scale: int = 16) -> Image.Image:
    c = np.array(cover_img.convert("RGB"))
    s = np.array(stego_img.convert("RGB"))
    diff = np.abs(s.astype(np.int16) - c.astype(np.int16)).astype(np.uint8)
    return to_image((diff * scale).clip(0, 255).astype(np.uint8))


def metrics_text(cover_img: Image.Image, stego_img: Image.Image) -> str:
    m = compute_metrics(cover_img, stego_img)
    return f"MSE: {m['MSE']:.2f} | PSNR: {m['PSNR']:.2f} dB | SSIM: {m['SSIM']:.4f}"


def make_blink_gif_path(cover: Image.Image, stego: Image.Image, duration_ms: int = 350) -> str:
    # Save a tiny 2-frame GIF to a temp path so Gradio can render it reliably
    frames = [cover.convert("RGB"), stego.convert("RGB")]
    tmp = NamedTemporaryFile(suffix=".gif", delete=False)
    frames[0].save(tmp, format="GIF", save_all=True, append_images=[frames[1]], loop=0, duration=duration_ms)
    tmp.flush(); tmp.close()
    return tmp.name


def read_file_bytes_from_path(path_or_str) -> tuple[str, bytes]:
    p = pathlib.Path(str(path_or_str))
    return p.name, p.read_bytes()


# ---------------- Annotation helpers ----------------
def _tile_hotspots(base_img: Image.Image, diff_amp: np.ndarray, grid: int = 16, top_k: int = 8) -> Image.Image:
    h, w = diff_amp.shape[:2]
    th, tw = max(1, h // grid), max(1, w // grid)
    scores = []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * th, gx * tw
            y1, x1 = min(h, y0 + th), min(w, x0 + tw)
            tile = diff_amp[y0:y1, x0:x1]
            score = float(tile.mean())
            scores.append((score, (x0, y0, x1, y1)))
    scores.sort(reverse=True, key=lambda t: t[0])
    picks = scores[:top_k]
    overlay = np.array(base_img.convert("RGB")).copy()
    for _, (x0, y0, x1, y1) in picks:
        overlay[y0:y1, [x0, min(x1-1, overlay.shape[1]-1)], :] = [255, 0, 0]
        overlay[[y0, min(y1-1, overlay.shape[0]-1)], x0:x1, :] = [255, 0, 0]
    return to_image(overlay)


def _lsb_parity_heatmap(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    lsb = (arr & 1).mean(axis=2)  # average across channels
    heat = (lsb * 255).astype(np.uint8)
    return to_image(heat)


def _top_energy_tiles(gray_img: np.ndarray, grid: int = 16, top_k: int = 8) -> list:
    h, w = gray_img.shape
    th, tw = max(1, h // grid), max(1, w // grid)
    scores = []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * th, gx * tw
            y1, x1 = min(h, y0 + th), min(w, x0 + tw)
            tile = gray_img[y0:y1, x0:x1]
            scores.append((float(tile.mean()), (x0, y0, x1, y1)))
    scores.sort(reverse=True, key=lambda t: t[0])
    return [bb for _, bb in scores[:top_k]]


def _annotate_boxes(gray_img: np.ndarray, boxes: list) -> Image.Image:
    vis = np.stack([gray_img, gray_img, gray_img], axis=2)
    vis = (vis * 255).astype(np.uint8)
    for (x0, y0, x1, y1) in boxes:
        vis[y0:y1, [x0, max(x1-1, 0)], :] = [0, 255, 0]
        vis[[y0, max(y1-1, 0)], x0:x1, :] = [0, 255, 0]
    return to_image(vis)


def encode_ui(cover_img: Image.Image, payload_path, password: str, keyed: bool, inverted: bool, encrypt: bool, fmt: str):
    if cover_img is None or payload_path is None:
        return None, None, None, None, "Please upload both a cover image and a payload file."

    cover_buf = BytesIO(); cover_img.save(cover_buf, format="PNG"); cover_bytes = cover_buf.getvalue()
    try:
        filename, payload_bytes = read_file_bytes_from_path(payload_path)
    except Exception as ex:
        return None, None, None, None, f"Error loading payload: {ex}"

    # Capacity check to avoid long-running failures
    try:
        cap = capacity_bytes(cover_img)
        if len(payload_bytes) > cap:
            return None, None, None, None, f"Error: Payload ({len(payload_bytes):,} B) exceeds capacity (~{cap:,} B)."
    except Exception:
        pass

    try:
        stego_bytes = hide_file_in_image(
            cover_img_bytes=cover_bytes,
            payload_bytes=payload_bytes,
            payload_filename=filename,
            password=password if (keyed or encrypt) else None,
            keyed=keyed,
            inverted=inverted,
            encrypt=encrypt,
            out_format=fmt,
        )
        # Try to build preview; if Pillow lacks codec, skip preview
        stego_img = None
        diff_amp = None
        blink_path = None
        hotspots = None
        lsb_heat = None
        try:
            preview = Image.open(BytesIO(stego_bytes)).convert("RGB")
            stego_img = preview
            diff_amp = np.array(diff_amp_image(cover_img, preview))
            blink_path = make_blink_gif_path(cover_img, preview)
            hotspots = tile_hotspots(preview, diff_amp, grid=16, top_k=8)
            lsb_heat = lsb_parity_heatmap(preview)
        except Exception:
            pass

        mtxt = metrics_text(cover_img, stego_img) if stego_img is not None else "Saved stego. Preview unavailable on this system."

        # Save exact stego bytes to a temp file for download (respects chosen format)
        suffix = ".png" if fmt.upper() == "PNG" else ".webp"
        tmp_out = NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_out.write(stego_bytes)
        tmp_out.flush(); tmp_out.close()
        stego_file_path = tmp_out.name

        hints = (
            "Highlighted red boxes: tiles with highest Cover→Stego differences.\n"
            "LSB heatmap: brighter = more 1-bits on average; uniformity shifts can indicate embedding."
        )

        return stego_img, to_image(diff_amp), blink_path, mtxt, stego_file_path, "Success", hotspots, lsb_heat, hints
    except Exception as e:
        return None, None, None, None, None, f"Error: {e}", None, None, None


def decode_ui(stego_img: Image.Image, password: str, invert_hint: bool, keyed_hint: bool):
    if stego_img is None:
        return None, None, "Please upload a stego image."
    buf = BytesIO(); stego_img.save(buf, format="PNG"); stego_bytes = buf.getvalue()
    try:
        fname, payload = extract_file_from_image(stego_bytes, password=password if (keyed_hint or password) else None, invert_hint=invert_hint, keyed_hint=keyed_hint)
        return fname, payload, "Success"
    except Exception as e:
        return None, None, f"Error: {e}"


with gr.Blocks(title="Stego Visual UI") as demo:
    gr.Markdown("# 🛡️ Visual Steganography — LSB + AES-GCM")
    with gr.Tab("Encode"):
    with gr.Row():
    cover_in = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="Cover image (lossless recommended)")
    payload_in = gr.File(label="Payload file", type="filepath")

# 🧩 Add payload preview right below
def preview_payload(file):
    import pathlib
    from PIL import Image
    if not file:
        return None, "No file selected."
    try:
        path = pathlib.Path(str(file))
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            img = Image.open(path)
            return img, f"Image payload: {path.name} ({img.width}×{img.height})"
        else:
            text_sample = path.read_text(errors="ignore")[:300]
            return None, f"Text payload preview ({path.name}):\n{text_sample}..."
    except Exception as e:
        return None, f"Cannot preview: {e}"

payload_preview, payload_info = gr.Image(label="Payload preview", visible=True), gr.Markdown()
payload_in.change(preview_payload, [payload_in], [payload_preview, payload_info])
        with gr.Row():
            password = gr.Textbox(type="password", label="Password (for keyed/encrypt)")
            keyed = gr.Checkbox(label="Keyed order", value=False)
            inverted = gr.Checkbox(label="Inverted LSB", value=False)
            encrypt = gr.Checkbox(label="Encrypt (AES-256-GCM)", value=False)
            fmt = gr.Radio(["PNG", "WEBP"], value="PNG", label="Output format")
        btn = gr.Button("🔐 Hide File")
        with gr.Row():
            stego_out = gr.Image(type="pil", label="Stego")
            diff_out = gr.Image(type="pil", label="Diff ×16")
            blink_out = gr.Image(label="Blink (Cover↔Stego)")
        stego_file = gr.File(label="Download stego (exact bytes)")
        with gr.Row():
            hot_out = gr.Image(type="pil", label="Hotspots (highest differences)")
            lsb_out = gr.Image(type="pil", label="LSB Parity Heatmap")
        hints = gr.Textbox(label="Highlights", interactive=False)
        metrics = gr.Textbox(label="Metrics", interactive=False)
        status = gr.Markdown()
        btn.click(
            encode_ui,
            [cover_in, payload_in, password, keyed, inverted, encrypt, fmt],
            [stego_out, diff_out, blink_out, metrics, stego_file, status, hot_out, lsb_out, hints],
        )

    with gr.Tab("Decode"):
        stego_path = gr.File(label="Stego image", type="filepath")
        with gr.Row():
            password2 = gr.Textbox(type="password", label="Password (if keyed/encrypted)")
            invert_hint = gr.Checkbox(label="Inverted LSB?", value=False)
            keyed_hint = gr.Checkbox(label="Keyed order?", value=False)
        with gr.Row():
            btn_diag = gr.Button("🔎 Diagnose Header")
            btn2 = gr.Button("🧠 Extract File")
        out_name = gr.Textbox(label="Filename", interactive=False)
        out_bytes = gr.File(label="Extracted file", interactive=False)
        status2 = gr.Markdown()
        header_info = gr.Textbox(label="Header Info", interactive=False)

        def decode_ui2(stego_file_path, password: str, invert_hint: bool, keyed_hint: bool):
            if stego_file_path is None:
                return None, None, "Please upload a stego image file."
            try:
                import pathlib
                raw_bytes = pathlib.Path(str(stego_file_path)).read_bytes()
            except Exception as e:
                return None, None, f"Error reading file: {e}"

            # Helper to write payload to a temp file for reliable download
            def _to_temp_file(filename: str, data: bytes) -> str:
                import os
                suffix = ("_" + filename) if filename else "_output.bin"
                tmp = NamedTemporaryFile(suffix="_" + filename if filename else "_output.bin", delete=False)
                tmp.write(data); tmp.flush(); tmp.close()
                return tmp.name

            # Primary attempt
            try:
                fname, payload = extract_file_from_image(
                    raw_bytes,
                    password=password if (keyed_hint or password) else None,
                    invert_hint=invert_hint,
                    keyed_hint=keyed_hint,
                )
                return fname, _to_temp_file(fname, payload), "Success"
            except Exception as e1:
                msg = str(e1)
                # Helpful guidance for common errors
                if "InvalidTag" in msg or "MAC check failed" in msg:
                    return None, None, "Error: Wrong password or corrupted encrypted payload."
                # Fallback attempts on any failure: try flipped inverted and toggled keyed combos
                alt_attempts = []
                alt_attempts.append((not invert_hint, keyed_hint))
                if password:
                    alt_attempts.append((invert_hint, not keyed_hint))
                    alt_attempts.append((not invert_hint, not keyed_hint))
                    alt_attempts.append((invert_hint, True))
                    alt_attempts.append((not invert_hint, True))
                for inv_alt, key_alt in alt_attempts:
                    try:
                        fname, payload = extract_file_from_image(
                            raw_bytes,
                            password=password if password else None,
                            invert_hint=inv_alt,
                            keyed_hint=key_alt,
                        )
                        note = " (auto-detected settings used)"
                        return fname, _to_temp_file(fname, payload), f"Success{note}"
                    except Exception:
                        continue
                return None, None, f"Error: {msg}"

        btn2.click(decode_ui2, [stego_path, password2, invert_hint, keyed_hint], [out_name, out_bytes, status2])

        # Diagnose header: read only the header fields without requiring exact keyed/inverted hints
        def diagnose_header(stego_file_path):
            if stego_file_path is None:
                return "Upload a stego image file first."
            try:
                import pathlib
                raw_bytes = pathlib.Path(str(stego_file_path)).read_bytes()
            except Exception as e:
                return f"Error reading file: {e}"
            try:
                # Header is written sequentially; hints don't matter for header parsing
                from stego.lsb import decode_lsb
                info, _ = decode_lsb(raw_bytes, inverted_hint=False, keyed=False, key_material=None)
                salt_hex = info['salt'].hex() if info.get('salt') else '(none)'
                iters = info.get('iters')
                enc = info.get('encrypted')
                plen = info.get('length')
                fn = info.get('filename')
                return (
                    f"Header OK.\n"
                    f"Encrypted: {enc}\n"
                    f"Filename: {fn}\n"
                    f"Length: {plen if plen is not None else 'unknown'} bytes\n"
                    f"Salt: {salt_hex}\n"
                    f"Iterations: {iters if iters is not None else 'unknown'}\n"
                    f"Hints: If decoding fails, try toggling Inverted/Keyed and ensure the same password."
                )
            except Exception as e:
                return f"No header found or corrupted: {e}"

        btn_diag.click(diagnose_header, [stego_path], [header_info])

    # ---------------- Quality & Stego Test ----------------
    with gr.Tab("Quality & Stego Test"):
        q_cov = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="Cover image")
        q_stg = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="Stego image")
        q_btn = gr.Button("📊 Analyze")
        q_metrics = gr.Textbox(label="Metrics (PSNR/SSIM/MSE)", interactive=False)
        q_hist = gr.Plot(label="Grayscale Histograms")
        q_chi = gr.Plot(label="Chi-square (normalized)")
        q_hot = gr.Image(type="pil", label="Hotspots (highest differences)")
        q_notes = gr.Textbox(label="Highlights", interactive=False)

        def quality_analyze(cov: Image.Image, stg: Image.Image):
            if cov is None or stg is None:
                return "Upload both images.", plt.figure(), plt.figure(), None, ""
            mtxt = metrics_text(cov, stg)

            cov_arr = np.array(cov.convert("RGB"))
            stg_arr = np.array(stg.convert("RGB"))
            cov_gray = np.mean(cov_arr, axis=2).astype(np.uint8)
            stg_gray = np.mean(stg_arr, axis=2).astype(np.uint8)
            cov_hist, _ = np.histogram(cov_gray, bins=256, range=(0, 256))
            stg_hist, _ = np.histogram(stg_gray, bins=256, range=(0, 256))

            chi_cov = chi_square_lsb(cov)
            chi_stg = chi_square_lsb(stg)

            fig_hist, axh = plt.subplots(figsize=(6,3))
            axh.plot(cov_hist, color='gray', alpha=.8, label='Cover')
            axh.plot(stg_hist, color='tab:blue', alpha=.8, label='Stego')
            axh.set_title('Grayscale Histogram')
            axh.legend()

            fig_chi, axc = plt.subplots(figsize=(4,3))
            axc.bar(['Cover','Stego'], [chi_cov['norm_stat'], chi_stg['norm_stat']], color=['gray','tab:blue'])
            axc.set_title('Chi-square (normalized)')

            # Hotspots from differences
            diff_amp = np.array(diff_amp_image(cov, stg))
            hot_img = tile_hotspots(Image.fromarray(stg_arr), diff_amp, grid=16, top_k=8)
            notes = (
                "Boxes mark tiles with the highest Cover→Stego differences.\n"
                "Histogram shape shifts and increased chi-square can indicate LSB embedding."
            )
            return mtxt, fig_hist, fig_chi, hot_img, notes

        q_btn.click(quality_analyze, [q_cov, q_stg], [q_metrics, q_hist, q_chi, q_hot, q_notes])

    # ---------------- Transforms (DCT & WHT) ----------------
    with gr.Tab("Transforms (DCT & WHT)"):
        t_img = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="Image for transform analysis")
        t_btn = gr.Button("🧮 Compute")
        t_dct = gr.Image(label="DCT Gray (log-mag)")
        t_wht = gr.Image(label="WHT magnitude (log)")
        t_annot = gr.Image(label="Annotated hotspots (DCT)")
        t_notes = gr.Textbox(label="Highlights", interactive=False)

        def transforms_compute(img: Image.Image):
            if img is None:
                return None, None, None, ""
            dct_gray, _, _, _ = dct_magnitude(img)
            wht_mag, _ = wht2(img)
            dct_img = (dct_gray * 255).astype(np.uint8)
            wht_img = (wht_mag * 255).astype(np.uint8)
            # Annotate top-energy tiles on DCT
            boxes = top_energy_tiles(dct_gray, grid=16, top_k=8)
            ann = annotate_boxes(dct_gray, boxes)
            notes = (
                "Green boxes mark high-energy spectral regions.\n"
                "Embedding can nudge energy in certain bands; compare cover vs stego images here."
            )
            return to_image(dct_img), to_image(wht_img), ann, notes

        t_btn.click(transforms_compute, [t_img], [t_dct, t_wht, t_annot, t_notes])


if __name__ == "__main__":
    demo.launch()

