# app/streamlit_app.py
from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


from metrics.analysis import compute_metrics, dct_magnitude, wht2, chi_square_lsb
import streamlit as st
from PIL import Image
from io import BytesIO

from stego.lsb import encode_lsb, decode_lsb, capacity_bytes
from crypto.pbkdf2_aesgcm import encrypt_aes_gcm, decrypt_aes_gcm, DEFAULT_ITERS


st.set_page_config(page_title="Stego App", page_icon="🛡️", layout="wide")

# ---------------- STYLE ----------------
st.markdown(
    """
    <style>
    .stApp {background-color: #0e1117;}
    .section-title {font-size: 1.2rem; font-weight: 700; margin-top: .75rem;}
    .pill {display: inline-block; padding: 2px 8px; border-radius: 999px; background: #1f2937; color: #e5e7eb; margin-right: 6px; font-size: .8rem;}
    .ok {background: #065f46 !important;}
    .warn {background: #78350f !important;}
    .err {background: #7f1d1d !important;}
    .card {background: #111827; border: 1px solid #1f2937; padding: .9rem; border-radius: .75rem;}
    .muted {color: #9ca3af; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Steganography — LSB + Keyed + Inverted + AES-GCM")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About")
    st.caption("Research-grade, reproducible steganography demo with LSB embedding, AES-256-GCM, and analysis.")
    st.subheader("Security Notes")
    st.markdown("- Uses PBKDF2-HMAC(SHA256) → AES-256-GCM with AAD bound to filename")
    st.markdown("- Header includes version, salt, iters, and exact payload length")
    st.markdown("- Lossless PNG/WEBP recommended for integrity")
    st.divider()
    st.subheader("Tips")
    st.markdown("- Prefer larger covers; keep payload small for imperceptibility")
    st.markdown("- Compare PSNR/SSIM cover vs stego in the Metrics tab")
    st.markdown("- Use Keyed + Encrypt for stronger adversarial robustness")

tab1, tab2 = st.tabs(["🧩 Encode", "🔍 Decode"])

# ---------------- ENCODE ----------------
with tab1:
    st.subheader("Hide a file inside an image")
    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='section-title'>Inputs</div>", unsafe_allow_html=True)
        cover_file = st.file_uploader("Cover image (PNG/WEBP lossless)", type=["png", "webp"])
        payload_file = st.file_uploader("File to hide (any type)")
        fmt = st.radio("Output format", ["PNG", "WEBP"], horizontal=True)
    with right:
        st.markdown("<div class='section-title'>Options</div>", unsafe_allow_html=True)
        inverted = st.checkbox("Inverted LSB", value=False)
        keyed = st.checkbox("Keyed pixel order", value=False)
        encrypt = st.checkbox("Encrypt payload (AES-256-GCM)", value=False)
        password = st.text_input("Password (required if keyed and/or encrypt)", type="password")

    st.markdown("<div class='section-title'>Preview & Capacity</div>", unsafe_allow_html=True)
    cprev, cinfo = st.columns([2, 1])

    cover_bytes = None
    cover_img = None
    if cover_file:
        try:
            cover_bytes = cover_file.getvalue()
            cover_img = Image.open(BytesIO(cover_bytes)).convert("RGB")
            st.image(cover_img, caption=f"Cover {cover_img.width}×{cover_img.height}", use_column_width=True)
            cap = capacity_bytes(cover_img)
            st.caption(f"Capacity: ~{cap:,} bytes (1 LSB/channel)")
        except Exception as e:
            st.error(f"Invalid cover image: {e}")
            cover_bytes = None
            cover_img = None

    # Capacity visualization
    with cinfo:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if cover_img and payload_file:
            payload_size = len(payload_file.getvalue())
            cap = capacity_bytes(cover_img)
            used_ratio = min(payload_size / max(cap, 1), 1.0)
            st.markdown("**Payload size**")
            st.text(f"{payload_size:,} bytes")
            st.markdown("**Estimated capacity use**")
            st.progress(used_ratio, text=f"{used_ratio*100:.1f}% of capacity")
            if used_ratio >= 1.0:
                st.error("Payload exceeds available capacity.")
            elif used_ratio > 0.8:
                st.warning("High utilization: visual artifacts more likely.")
        else:
            st.markdown("<span class='muted'>Upload cover and payload to see capacity usage.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    can_encode = bool(cover_bytes) and bool(payload_file)
    if (keyed or encrypt) and not password:
        can_encode = False

    go = st.button("🔐 Hide File", type="primary", disabled=not can_encode)

    if go:
        if not cover_bytes or not payload_file:
            st.warning("Upload both a cover image and a file.")
        elif (keyed or encrypt) and not password:
            st.warning("Password is required when Keyed or Encrypt is enabled.")
        else:
            try:
                payload = payload_file.getvalue()
                filename = payload_file.name

                # Encrypt if requested
                encrypted_flag = False
                salt = None
                iters = None
                blob = payload
                if encrypt:
                    blob, salt, iters = encrypt_aes_gcm(payload, password, DEFAULT_ITERS, aad=filename.encode("utf-8"))
                    encrypted_flag = True

                # Keyed pixel order seed = password bytes (must match decode)
                key_material = password.encode("utf-8") if keyed else None

                stego_bytes = encode_lsb(
                    cover_img_bytes=cover_bytes,
                    filename=filename,
                    payload=blob,
                    encrypted=encrypted_flag,
                    inverted=inverted,
                    keyed=keyed,
                    out_format=fmt,
                    key_material=key_material,
                    salt=salt,
                    iters=iters,
                )

                st.success("✅ File successfully hidden!")
                cdl, cviz = st.columns([1, 2])
                with cdl:
                    st.download_button(
                        "⬇️ Download Stego Image",
                        data=stego_bytes,
                        file_name=f"stego.{fmt.lower()}",
                        mime=f"image/{fmt.lower()}",
                    )
                    st.markdown(
                        f"<span class='pill ok'>Encrypted: {'Yes' if encrypted_flag else 'No'}</span>"
                        f"<span class='pill'>{'Keyed' if keyed else 'Sequential'}</span>"
                        f"<span class='pill'>{fmt}</span>",
                        unsafe_allow_html=True,
                    )
                with cviz:
                    try:
                        stego_img = Image.open(BytesIO(stego_bytes)).convert("RGB")
                        colA, colB = st.columns(2)
                        with colA:
                            st.image(cover_img, caption="Cover", use_column_width=True)
                        with colB:
                            st.image(stego_img, caption="Stego", use_column_width=True)
                        m = compute_metrics(cover_img, stego_img)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("PSNR (dB)", f"{m['PSNR']:.2f}")
                        m2.metric("SSIM", f"{m['SSIM']:.4f}")
                        m3.metric("MSE", f"{m['MSE']:.2f}")
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"Embedding failed: {e}")

# ---------------- DECODE ----------------
with tab2:
    st.subheader("Extract the hidden file")
    st.markdown("<div class='section-title'>Inputs</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        stego_file = st.file_uploader("Stego image (PNG/WEBP lossless)", type=["png", "webp"], key="stego_up")
    with col2:
        invert_hint = st.checkbox("Inverted LSB?", value=False)
        keyed_hint = st.checkbox("Keyed order?", value=False)
    with col3:
        password2 = st.text_input("Password", type="password", key="pw2")

    can_extract = bool(stego_file) and (not keyed_hint or (keyed_hint and password2))
    go2 = st.button("🧠 Extract File", type="primary", disabled=not can_extract)

    stego_bytes = None
    if stego_file:
        try:
            stego_bytes = stego_file.getvalue()
            prev = Image.open(BytesIO(stego_bytes)).convert("RGB")
            st.image(prev, caption=f"Stego {prev.width}×{prev.height}", use_column_width=True)
        except Exception as e:
            st.error(f"Invalid stego image: {e}")
            stego_bytes = None

    if go2:
        if not stego_bytes:
            st.warning("Upload a stego image.")
        elif keyed_hint and not password2:
            st.warning("Password is required when Keyed is ON.")
        else:
            try:
                key_material = password2.encode("utf-8") if keyed_hint else None
                info, content = decode_lsb(
                    stego_bytes,
                    inverted_hint=invert_hint,
                    keyed=keyed_hint,
                    key_material=key_material,
                )

                # If encrypted, decrypt using header salt/iters
                if info["encrypted"]:
                    if not password2:
                        st.error("This payload is encrypted. Provide the password to decrypt.")
                        st.stop()
                    if info["salt"] is None or info["iters"] is None:
                        st.error("Encrypted payload missing salt/iters (corrupt header?).")
                        st.stop()
                    try:
                        content = decrypt_aes_gcm(content, password2, info["salt"], info["iters"], aad=info.get("filename","output.bin").encode("utf-8"))
                    except Exception as ex:
                        st.error(f"Decryption failed: {ex}")
                        st.stop()

                st.success(f"✅ Extracted file: {info.get('filename','output.bin')}")
                cdl2, meta2 = st.columns([1, 2])
                with cdl2:
                    st.download_button(
                        "⬇️ Download Extracted File",
                        data=content,
                        file_name=info.get("filename", "output.bin"),
                    )
                with meta2:
                    st.markdown(
                        f"<span class='pill {'ok' if info.get('encrypted') else ''}'>Encrypted: { 'Yes' if info.get('encrypted') else 'No'}</span>"
                        f"<span class='pill'>Length: {info.get('length') if info.get('length') else 'unknown'}</span>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Extraction failed: {e}")


tabA, tabB, tabC = st.tabs(["📊 Quality & Stego Test", "🧮 DCT Analysis", "⚡ Walsh–Hadamard"])

with tabA:
    st.subheader("Quality metrics & Chi-square steganalysis")
    col1, col2 = st.columns(2)
    with col1:
        cover_for_eval = st.file_uploader("Cover image (for metrics)", type=["png","webp","jpg","jpeg"], key="cov_eval")
    with col2:
        stego_for_eval = st.file_uploader("Stego image (for metrics)", type=["png","webp","jpg","jpeg"], key="stg_eval")

    if cover_for_eval and stego_for_eval:
        try:
            cov = Image.open(cover_for_eval).convert("RGB")
            stg = Image.open(stego_for_eval).convert("RGB")
            if cov.size != stg.size:
                st.warning(f"Images differ in size: {cov.size} vs {stg.size}. Metrics need same size.")
            else:
                m = compute_metrics(cov, stg)
                st.success(f"MSE: {m['MSE']:.2f} | PSNR: {m['PSNR']:.2f} dB | SSIM: {m['SSIM']:.4f}")
        except Exception as e:
            st.error(f"Metric error: {e}")

    st.divider()
    st.caption("Chi-square LSB test (on a single image) — higher stats can indicate suspicious LSB patterns.")
    chi_file = st.file_uploader("Choose an image (cover or stego)", type=["png","webp","jpg","jpeg"], key="chi")
    if chi_file:
        try:
            im = Image.open(chi_file).convert("RGB")
            res = chi_square_lsb(im)
            st.info(f"Chi-square stat: {res['chi2_stat']:.1f}  |  normalized: {res['norm_stat']:.3f}")
            st.caption("Rule of thumb: compare this stat for your cover vs stego on the same base image—"
                       "stego usually lifts the statistic due to LSB distortions.")
        except Exception as e:
            st.error(f"Chi-square error: {e}")

with tabB:
    st.subheader("2D DCT magnitude (log-scaled)")
    dct_img = st.file_uploader("Image for DCT analysis", type=["png","webp","jpg","jpeg"], key="dct_up")
    if dct_img:
        try:
            im = Image.open(dct_img).convert("RGB")
            mag_gray, mag_r, mag_g, mag_b = dct_magnitude(im)
            st.image([im, (mag_gray*255).astype("uint8")], caption=["Input", "DCT | Gray (log-mag)"], use_column_width=True)
            st.image([(mag_r*255).astype("uint8"),
                      (mag_g*255).astype("uint8"),
                      (mag_b*255).astype("uint8")],
                     caption=["DCT | R", "DCT | G", "DCT | B"], use_column_width=True)
            st.caption("Brighter = higher frequency energy. Compare cover vs stego to see subtle spectral shifts.")
        except Exception as e:
            st.error(f"DCT error: {e}")

with tabC:
    st.subheader("Walsh–Hadamard Transform (WHT) magnitude (log-scaled)")
    wht_img = st.file_uploader("Image for WHT analysis", type=["png","webp","jpg","jpeg"], key="wht_up")
    if wht_img:
        try:
            im = Image.open(wht_img).convert("RGB")
            mag, sz = wht2(im)
            st.image([im, (mag*255).astype("uint8")],
                     caption=[f"Input ({im.size[0]}×{im.size[1]})",
                              f"WHT | Gray (padded to {sz[0]}×{sz[1]})"],
                     use_column_width=True)
            st.caption("WHT is binary ±1 basis; useful for fast, coarse spectral analysis. "
                       "Again, compare cover vs stego.")
        except Exception as e:
            st.error(f"WHT error: {e}")