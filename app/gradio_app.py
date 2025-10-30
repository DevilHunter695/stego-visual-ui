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
# Normalize gr.File input which can be a str path or a dict in some versions
def _normalize_file_input(v) -> Optional[str]:
    try:
        if v is None:
            return None
        # Already a path-like
        if isinstance(v, (str, pathlib.Path)):
            return str(v)
        # Gradio may return a dict with 'name'
        if isinstance(v, dict):
            name = v.get('name') or v.get('path') or v.get('file')
            return str(name) if name else None
    except Exception:
        pass
    return None



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

        diff_img = to_image(diff_amp) if diff_amp is not None else None
        return stego_img, diff_img, blink_path, mtxt, stego_file_path, "Success", hotspots, lsb_heat, hints
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
            cover_in = gr.Image(type="pil", image_mode="RGB", sources=["upload"], label="Cover image (lossless recommended)", elem_classes="cover-image")
            with gr.Column():
                payload_in = gr.File(label="Payload file", type="filepath", elem_classes="payload-file")
                payload_file_preview = gr.Image(label="Payload Preview", visible=False, elem_classes="sm-image")
        with gr.Row():
            password = gr.Textbox(type="password", label="Password (for keyed/encrypt)", elem_classes="sm-textbox")
            keyed = gr.Checkbox(label="Keyed order", value=False, elem_classes="sm-checkbox")
            inverted = gr.Checkbox(label="Inverted LSB", value=False, elem_classes="sm-checkbox")
            encrypt = gr.Checkbox(label="Encrypt (AES-256-GCM)", value=False, elem_classes="sm-checkbox")
            fmt = gr.Radio(["PNG", "WEBP"], value="PNG", label="Output format", elem_classes="sm-radio")
        btn = gr.Button("🔐 Hide File", elem_classes="sm-button")
        with gr.Row():
            stego_out = gr.Image(type="pil", label="Stego", elem_classes="sm-image")
            diff_out = gr.Image(type="pil", label="Diff ×16", elem_classes="sm-image")
            blink_out = gr.Image(label="Blink (Cover↔Stego)", elem_classes="sm-image")
        stego_file = gr.File(label="Download stego (exact bytes)", elem_classes="sm-file")
        with gr.Row():
            hot_out = gr.Image(type="pil", label="Hotspots (highest differences)", elem_classes="sm-image")
            lsb_out = gr.Image(type="pil", label="LSB Parity Heatmap", elem_classes="sm-image")
        # Preview extracted payload as image if possible (for download stego, not upload)
        payload_preview = gr.Image(label="Payload Preview", visible=False, elem_classes="sm-image")
        hints = gr.Markdown(elem_classes="sm-md")
        metrics = gr.Markdown(elem_classes="sm-md")
        status = gr.Markdown()

        # Update encode_ui return/output to include payload_preview if appropriate
        def encode_ui_payload_preview_wrap(*args):
            out = encode_ui(*args)
            stego_img, diff_img, blink, mtxt, stego_file_path, status_msg, hotspots, lsb_map, hints_ = out
            # Update preview if the payload is image
            preview_img = None
            payload_path = args[1]
            try:
                fname = str(payload_path)
                import os
                from PIL import Image
                img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]
                if fname and any(fname.lower().endswith(e) for e in img_exts):
                    img = Image.open(fname)
                    preview_img = img
            except Exception:
                preview_img = None
            return stego_img, diff_img, blink, mtxt, stego_file_path, status_msg, hotspots, lsb_map, hints_, preview_img

        btn.click(
            encode_ui_payload_preview_wrap,
            [cover_in, payload_in, password, keyed, inverted, encrypt, fmt],
            [stego_out, diff_out, blink_out, metrics, stego_file, status, hot_out, lsb_out, hints, payload_preview],
        )
        payload_preview.change(lambda x: gr.update(visible=bool(x)), inputs=payload_preview, outputs=payload_preview)

        # Show payload file thumbnail when uploading (if image)
        def payload_file_preview_fn(payload_path):
            try:
                fname = str(payload_path)
                img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]
                if fname and any(fname.lower().endswith(e) for e in img_exts):
                    from PIL import Image
                    img = Image.open(fname)
                    return img
            except Exception:
                pass
            return None
        payload_in.change(payload_file_preview_fn, inputs=payload_in, outputs=payload_file_preview)
        payload_file_preview.change(lambda x: gr.update(visible=bool(x)), inputs=payload_file_preview, outputs=payload_file_preview)

    with gr.Tab("Decode"):
        with gr.Row():
            with gr.Column():
                stego_path = gr.File(label="Stego image", type="filepath", elem_classes="sm-file")
                stego_file_preview = gr.Image(label="Stego Preview", visible=False, elem_classes="sm-image")
            with gr.Column():
                pass # leave for symmetry
        with gr.Row():
            password2 = gr.Textbox(type="password", label="Password (if keyed/encrypted)", elem_classes="sm-textbox")
            invert_hint = gr.Checkbox(label="Inverted LSB?", value=False, elem_classes="sm-checkbox")
            keyed_hint = gr.Checkbox(label="Keyed order?", value=False, elem_classes="sm-checkbox")

        with gr.Row():
            btn_diag = gr.Button("🔎 Diagnose Header", elem_classes="sm-button")
            btn2 = gr.Button("🧠 Extract File", elem_classes="sm-button")

        out_name = gr.Textbox(label="Filename", interactive=False, elem_classes="sm-textbox")
        with gr.Row():
            out_bytes = gr.File(label="Extracted file", interactive=False, elem_classes="sm-file")
            extracted_file_preview = gr.Image(label="Extracted Preview", visible=False, elem_classes="sm-image")
        decode_payload_preview = gr.Image(label="Payload Preview", visible=False, elem_classes="sm-image")
        status2 = gr.Markdown()
        header_info = gr.Markdown(elem_classes="sm-md")

        # Helper function for decode
        def decode_ui2(stego_file_path, password: str, invert_hint: bool, keyed_hint: bool):
            p = _normalize_file_input(stego_file_path)
            if not p:
                return None, None, "Please upload a stego image file."
            try:
                raw_bytes = pathlib.Path(str(p)).read_bytes()
            except Exception as e:
                return None, None, f"Error reading file: {e}"

            def _to_temp_file(filename: str, data: bytes) -> str:
                tmp = NamedTemporaryFile(suffix="_" + (filename or "output.bin"), delete=False)
                tmp.write(data); tmp.flush(); tmp.close()
                return tmp.name

            try:
                fname, payload = extract_file_from_image(
                    raw_bytes,
                    password=password if (keyed_hint or password) else None,
                    invert_hint=invert_hint,
                    keyed_hint=keyed_hint,
                )
                path = _to_temp_file(fname, payload)
                return fname, path, "✅ Success"
            except Exception as e:
                return None, None, f"❌ Error: {e}"

        # Add image preview logic for extracted file
        def decode_ui2_with_preview(stego_file_path, password: str, invert_hint: bool, keyed_hint: bool):
            fname, file_path, msg = decode_ui2(stego_file_path, password, invert_hint, keyed_hint)
            img = None
            try:
                if file_path and any(file_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]):
                    img = Image.open(file_path)
            except Exception:
                img = None
            return fname, file_path, msg, img

        btn2.click(
            decode_ui2_with_preview,
            [stego_path, password2, invert_hint, keyed_hint],
            [out_name, out_bytes, status2, extracted_file_preview],
        )
        extracted_file_preview.change(lambda x: gr.update(visible=bool(x)), inputs=extracted_file_preview, outputs=extracted_file_preview)

        # Show stego file thumbnail when uploading (if image)
        def stego_file_preview_fn(stego_file_path):
            try:
                fname = _normalize_file_input(stego_file_path)
                img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]
                if fname and any(fname.lower().endswith(e) for e in img_exts):
                    from PIL import Image
                    img = Image.open(fname)
                    return img
            except Exception:
                pass
            return None
        stego_path.change(stego_file_preview_fn, inputs=stego_path, outputs=stego_file_preview)
        stego_file_preview.change(lambda x: gr.update(visible=bool(x)), inputs=stego_file_preview, outputs=stego_file_preview)

        # Diagnose header
        def diagnose_header(stego_file_path):
            p = _normalize_file_input(stego_file_path)
            if not p:
                return "Upload a stego image file first."
            try:
                raw_bytes = pathlib.Path(str(p)).read_bytes()
            except Exception as e:
                return f"Error reading file: {e}"
            try:
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
        q_metrics = gr.Markdown()
        q_hist = gr.Plot(label="Grayscale Histograms")
        q_chi = gr.Plot(label="Chi-square (normalized)")
        q_hot = gr.Image(type="pil", label="Hotspots (highest differences)")
        q_notes = gr.Markdown()

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
        t_notes = gr.Markdown()

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

# At file end, inject the CSS for a tighter, more visually balanced look
css_custom = '''
/* ---- Global spacing ---- */
.gradio-container { --gap-lg: 10px; --gap-md: 8px; --gap-sm: 6px; }
.gradio-container .gr-row { gap: var(--gap-lg); align-items: flex-start; }
.gradio-container .gr-column { gap: var(--gap-md); }
/* Slightly compact base text rhythm */
.gradio-container * { line-height: 1.25; }

/* ---- Images (consistent thumbnails) ---- */
.sm-image { width: 200px; margin: 3px; }
.sm-image img, .sm-image canvas { max-height: 160px; width: 100%; height: auto; border-radius: 8px; object-fit: contain; }
.cover-image { width: 220px; }
.cover-image img, .cover-image canvas { max-height: 180px; width: 100%; height: auto; border-radius: 8px; object-fit: contain; }

/* ---- File inputs and general widths ---- */
.payload-file, .sm-file { font-size: 0.96em; padding: 4px 8px; width: 240px; }

/* ---- Text inputs ---- */
.sm-textbox { min-height: 40px; font-size: 1em; border-radius: 6px; }

/* ---- Toggles ---- */
.sm-checkbox input[type=checkbox] { width: 17px; height: 17px; }
.sm-radio input[type=radio] { width: 16px; height: 16px; }

/* ---- Buttons ---- */
.sm-button { padding: 6px 14px; font-size: 1em; border-radius: 7px; }

/* ---- Clean, auto-sized Markdown blocks ---- */
.sm-md { padding: 6px 8px; background: #fafafa; border-radius: 7px; border: 1px solid #eee; font-size: 1.02em; }
.sm-md p { margin: 4px 0; }

/* ---- Plots & generic media ---- */
.gradio-container .gr-plot, .gradio-container .gr-plot * { max-width: 100%; }

/* ---- Row compaction for preview rows ---- */
.gradio-container .gr-row .sm-image + .sm-image { margin-left: 2px; }

/* ---- Responsive tweaks ---- */
@media (max-width: 960px) {
  .sm-image { width: 100%; }
  .cover-image { width: 100%; }
  .payload-file, .sm-file { width: 100%; }
}
'''
demo.css += css_custom


