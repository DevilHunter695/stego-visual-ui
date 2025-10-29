# app/cli.py
from __future__ import annotations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import argparse, sys as _sys
from pathlib import Path
from PIL import Image
from stego.lsb import encode_lsb, decode_lsb, capacity_bytes

# If you don't have crypto.pbkdf2_aesgcm, you can temporarily disable encrypt flags.
try:
    from crypto.pbkdf2_aesgcm import encrypt_aes_gcm, decrypt_aes_gcm, DEFAULT_ITERS, derive_key
    HAVE_CRYPTO = True
except Exception:
    HAVE_CRYPTO = False

def cmd_capacity(a):
    img = Image.open(a.image).convert("RGB")
    print(capacity_bytes(img))

def cmd_hide(a):
    cover = Path(a.cover).read_bytes()
    payload = Path(a.infile).read_bytes()
    filename = Path(a.infile).name

    encrypted = False
    salt = None
    iters = None

    # ENCRYPT (optional)
    if a.encrypt:
        if not HAVE_CRYPTO:
            print("[err] --encrypt requested but crypto module not available.", file=_sys.stderr)
            _sys.exit(2)
        if not a.password:
            print("[err] --password is required when --encrypt is used.", file=_sys.stderr)
            _sys.exit(2)
        blob, salt, iters = encrypt_aes_gcm(payload, a.password, DEFAULT_ITERS, aad=filename.encode("utf-8"))
        payload = blob
        encrypted = True

    # KEYED ORDER (seed) — use ONLY the password bytes so decode can reproduce it
    key_material = None
    if a.keyed:
        if not a.password:
            print("[err] --password is required when --keyed is used.", file=_sys.stderr)
            _sys.exit(2)
        key_material = a.password.encode("utf-8")

    out_bytes = encode_lsb(
        cover_img_bytes=cover,
        filename=filename,
        payload=payload,
        encrypted=encrypted,
        inverted=a.invert,
        keyed=a.keyed,
        out_format=a.format,
        key_material=key_material,
        salt=salt,
        iters=iters,
    )
    Path(a.out).write_bytes(out_bytes)
    print(f"[ok] wrote → {a.out}")

def cmd_extract(a):
    stego = Path(a.stego).read_bytes()

    # same keyed seed as encode: ONLY password bytes
    key_material = a.password.encode("utf-8") if a.password else None
    info, payload = decode_lsb(
        stego,
        inverted_hint=a.invert,
        keyed=bool(a.password),
        key_material=key_material
    )

    data = payload
    if info["encrypted"]:
        if not HAVE_CRYPTO:
            print("[err] encrypted payload detected but crypto module not available.", file=_sys.stderr)
            _sys.exit(2)
        if not a.password:
            print("[err] encrypted payload detected. Provide --password.", file=_sys.stderr)
            _sys.exit(2)
        if info.get("salt") is None or info.get("iters") is None:
            print("[warn] ENC header present but missing salt/iters. Cannot decrypt.")
        else:
            try:
                data = decrypt_aes_gcm(payload, a.password, info["salt"], info["iters"], aad=info.get("filename","output.bin").encode("utf-8"))
            except Exception as ex:
                print(f"[err] Decryption failed: {ex}", file=_sys.stderr)
                _sys.exit(2)

    outname = a.out or info["filename"]
    Path(outname).write_bytes(data)
    print(f"[ok] extracted → {outname}")

def main():
    p = argparse.ArgumentParser(description="Stego CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capacity")
    pc.add_argument("--image", required=True)
    pc.set_defaults(func=cmd_capacity)

    ph = sub.add_parser("hide")
    ph.add_argument("--cover", required=True)
    ph.add_argument("--infile", required=True)
    ph.add_argument("--out", required=True)
    ph.add_argument("--format", choices=["PNG","WEBP"], default="PNG")
    ph.add_argument("--invert", action="store_true")
    ph.add_argument("--keyed", action="store_true")
    ph.add_argument("--encrypt", action="store_true")
    ph.add_argument("--password", help="Password for keyed (and encrypt if enabled)")
    p.set_defaults
    ph.set_defaults(func=cmd_hide)

    pe = sub.add_parser("extract")
    pe.add_argument("--stego", required=True)
    pe.add_argument("--invert", action="store_true")
    pe.add_argument("--password", help="Password if keyed/encrypted was used")
    pe.add_argument("--out", help="Output file (default: embedded filename)")
    pe.set_defaults(func=cmd_extract)

    a = p.parse_args()
    a.func(a)

if __name__ == "__main__":
    main()