from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

from stego.lsb import encode_lsb, decode_lsb
from crypto.pbkdf2_aesgcm import encrypt_aes_gcm, decrypt_aes_gcm, DEFAULT_ITERS


def hide_file_in_image(
    cover_img_bytes: bytes,
    payload_bytes: bytes,
    payload_filename: str,
    password: Optional[str] = None,
    keyed: bool = False,
    inverted: bool = False,
    encrypt: bool = False,
    out_format: str = "PNG",
) -> bytes:
    """High-level API: returns stego image bytes.
    encrypt → AES-GCM with AAD=filename; keyed → password-seeded order; inverted → flip payload bits.
    """
    blob = payload_bytes
    salt = None
    iters = None
    enc_flag = False
    if encrypt:
        if not password:
            raise ValueError("Password required when encrypt=True.")
        blob, salt, iters = encrypt_aes_gcm(payload_bytes, password, DEFAULT_ITERS, aad=payload_filename.encode("utf-8"))
        enc_flag = True

    key_material = password.encode("utf-8") if keyed and password else None
    return encode_lsb(
        cover_img_bytes=cover_img_bytes,
        filename=payload_filename,
        payload=blob,
        encrypted=enc_flag,
        inverted=inverted,
        keyed=keyed,
        out_format=out_format,
        key_material=key_material,
        salt=salt,
        iters=iters,
    )


def extract_file_from_image(
    stego_img_bytes: bytes,
    password: Optional[str] = None,
    invert_hint: bool = False,
    keyed_hint: bool = False,
) -> Tuple[str, bytes]:
    """High-level API: returns (filename, payload bytes). Decrypts if needed.
    Uses filename as AAD for AES-GCM integrity binding.
    """
    key_material = password.encode("utf-8") if keyed_hint and password else None
    info, payload = decode_lsb(
        stego_img_bytes,
        inverted_hint=invert_hint,
        keyed=keyed_hint,
        key_material=key_material,
    )

    filename = info.get("filename", "output.bin")
    if info.get("encrypted"):
        if not password:
            raise ValueError("This payload is encrypted. Provide a password.")
        if info.get("salt") is None or info.get("iters") is None:
            raise ValueError("Encrypted payload missing salt/iters.")
        payload = decrypt_aes_gcm(payload, password, info["salt"], info["iters"], aad=filename.encode("utf-8"))
    return filename, payload


