from __future__ import annotations
import io, hashlib, random, binascii, math
from typing import Optional, Tuple, List, Dict
from PIL import Image


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def capacity_bytes(img: Image.Image) -> int:
    """Return total payload capacity (bytes) assuming 1 LSB per RGB channel."""
    return (img.width * img.height * 3) // 8


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte_val in data:
        for bit_index in range(8):
            bits.append((byte_val >> (7 - bit_index)) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        assembled = 0
        for j in range(8):
            if i + j < len(bits):
                assembled = (assembled << 1) | (1 if bits[i + j] else 0)
        out.append(assembled)
    return bytes(out)


# -------------------------------------------------
# Header helpers
# -------------------------------------------------
def _build_header(encrypted: bool, filename: str, payload_len: int, salt: Optional[bytes], iters: Optional[int]) -> str:
    if encrypted:
        salt_hex = binascii.hexlify(salt or b"").decode()
        return f"STG1:ENC:{filename}|{salt_hex}|{iters or 0}|{payload_len}|||"
    return f"STG1:PLA:{filename}|{payload_len}|||"


def _parse_header(header: str) -> Dict[str, Optional[object]]:
    """Return dict with keys: filename, encrypted, salt, iters, payload_len.
    Supports STG1 format and legacy PLA:/ENC: headers.
    """
    result: Dict[str, Optional[object]] = {
        "filename": "output.bin",
        "encrypted": False,
        "salt": None,
        "iters": None,
        "payload_len": None,
    }
    if header.startswith("STG1:PLA:"):
        core = header[len("STG1:PLA:"):-3]
        parts = core.split("|")
        result["filename"] = parts[0]
        result["payload_len"] = int(parts[1]) if len(parts) > 1 and parts[1] else None
        result["encrypted"] = False
        return result
    if header.startswith("STG1:ENC:"):
        core = header[len("STG1:ENC:"):-3]
        parts = core.split("|")
        result["filename"] = parts[0]
        salt_hex = parts[1] if len(parts) > 1 else ""
        iters_str = parts[2] if len(parts) > 2 else ""
        plen_str = parts[3] if len(parts) > 3 else ""
        result["salt"] = binascii.unhexlify(salt_hex) if salt_hex else None
        result["iters"] = int(iters_str or "0")
        result["payload_len"] = int(plen_str) if plen_str else None
        result["encrypted"] = True
        return result
    # Legacy headers
    if header.startswith("PLA:"):
        result["filename"] = header[4:-3]
        result["encrypted"] = False
        return result
    if header.startswith("ENC:"):
        core = header[4:-3]
        parts = core.split("|")
        result["filename"] = parts[0]
        salt_hex = parts[1] if len(parts) > 1 else ""
        iters_str = parts[2] if len(parts) > 2 else ""
        result["salt"] = binascii.unhexlify(salt_hex) if salt_hex else None
        result["iters"] = int(iters_str or "0")
        result["encrypted"] = True
        return result
    raise ValueError("Invalid header format.")


def _shuffled_indices(num_pixels: int, keyed: bool, key_material: Optional[bytes]) -> list:
    indices = list(range(num_pixels))
    if keyed and key_material:
        seed = int.from_bytes(hashlib.sha256(key_material).digest(), "big")
        random.Random(seed).shuffle(indices)
    return indices


# -------------------------------------------------
# ENCODE
# -------------------------------------------------
def encode_lsb(
    cover_img_bytes: bytes,
    filename: str,
    payload: bytes,
    encrypted: bool = False,
    inverted: bool = False,
    keyed: bool = False,
    out_format: str = "PNG",
    key_material: Optional[bytes] = None,
    salt: Optional[bytes] = None,
    iters: Optional[int] = None,
) -> bytes:
    """Embed payload bytes into image via LSB."""
    img = Image.open(io.BytesIO(cover_img_bytes)).convert("RGB")
    pixels = list(img.getdata())

    # ----- header -----
    payload_len = len(payload)
    header_str = _build_header(encrypted, filename, payload_len, salt, iters)

    header_bits = _bytes_to_bits(header_str.encode("utf-8"))
    payload_bits = _bytes_to_bits(payload)

    indices = _shuffled_indices(len(pixels), keyed, key_material)

    if inverted:
        payload_bits = [1 - b for b in payload_bits]

    total_bits = len(header_bits) + len(payload_bits)
    if total_bits > len(pixels) * 3:
        raise ValueError("Payload too large for cover image")

    new_pixels = pixels.copy()
    header_bit_index = 0

    # write header sequentially
    for pos in range(len(pixels)):
        if header_bit_index >= len(header_bits):
            break
        r, g, b = new_pixels[pos]
        if header_bit_index < len(header_bits):
            r = (r & 0xFE) | header_bits[header_bit_index]
            header_bit_index += 1
        if header_bit_index < len(header_bits):
            g = (g & 0xFE) | header_bits[header_bit_index]
            header_bit_index += 1
        if header_bit_index < len(header_bits):
            b = (b & 0xFE) | header_bits[header_bit_index]
            header_bit_index += 1
        new_pixels[pos] = (r, g, b)

    header_pixels_used = math.ceil(len(header_bits) / 3)

    # write payload (keyed order)
    payload_bit_index = 0
    for pos in indices:
        if pos < header_pixels_used:
            continue
        if payload_bit_index >= len(payload_bits):
            break
        r, g, b = new_pixels[pos]
        if payload_bit_index < len(payload_bits):
            r = (r & 0xFE) | payload_bits[payload_bit_index]
            payload_bit_index += 1
        if payload_bit_index < len(payload_bits):
            g = (g & 0xFE) | payload_bits[payload_bit_index]
            payload_bit_index += 1
        if payload_bit_index < len(payload_bits):
            b = (b & 0xFE) | payload_bits[payload_bit_index]
            payload_bit_index += 1
        new_pixels[pos] = (r, g, b)

    img.putdata(new_pixels)
    out = io.BytesIO()
    if out_format.upper() == "WEBP":
        img.save(out, format="WEBP", lossless=True, quality=100, method=6)
    else:
        img.save(out, format=out_format)
    return out.getvalue()


# -------------------------------------------------
# DECODE
# -------------------------------------------------
def decode_lsb(
    stego_bytes: bytes,
    inverted_hint: bool = False,
    keyed: bool = False,
    key_material: Optional[bytes] = None,
) -> Tuple[dict, bytes]:
    """Extract embedded payload."""
    img = Image.open(io.BytesIO(stego_bytes)).convert("RGB")
    pixels = list(img.getdata())

    # read header until delimiter found or until safe cap
    probe_bits: List[int] = []
    delimiter = b"|||"
    max_header_bytes = 65536  # very generous safety cap for header
    found = False
    for (r, g, b) in pixels:
        probe_bits += [r & 1, g & 1, b & 1]
        if len(probe_bits) % 24 == 0:  # every 3 bytes added, check
            probe_bytes = _bits_to_bytes(probe_bits)
            pos = probe_bytes.find(delimiter)
            if pos >= 0:
                header_len_bytes = pos + len(delimiter)
                header = probe_bytes[:header_len_bytes].decode("utf-8", errors="ignore")
                found = True
                break
            if len(probe_bytes) > max_header_bytes:
                break
    if not found:
        # final attempt if delimiter appears right after last chunk
        probe_bytes = _bits_to_bytes(probe_bits)
        pos = probe_bytes.find(delimiter)
        if pos < 0:
            raise ValueError("Header not found or corrupted.")
        header_len_bytes = pos + len(delimiter)
        header = probe_bytes[:header_len_bytes].decode("utf-8", errors="ignore")

    # parse header (supports STG1 + legacy)
    try:
        hdr = _parse_header(header)
    except Exception:
        raise ValueError("Corrupt or invalid header.")
    filename = hdr["filename"]  # type: ignore
    encrypted = bool(hdr["encrypted"])  # type: ignore
    salt = hdr["salt"]  # type: ignore
    iters = hdr["iters"]  # type: ignore
    payload_len = hdr["payload_len"]  # type: ignore

    header_len_bits = header_len_bytes * 8
    header_pixels_used = math.ceil(header_len_bits / 3)

    # keyed order
    indices = _shuffled_indices(len(pixels), keyed, key_material)

    # read payload
    payload_bits = []
    expected_bits = None if payload_len is None else int(payload_len) * 8
    for pos in indices:
        if pos < header_pixels_used:
            continue
        if expected_bits is not None and len(payload_bits) >= expected_bits:
            break
        r, g, b = pixels[pos]
        if expected_bits is None or len(payload_bits) < expected_bits: payload_bits.append(r & 1)
        if expected_bits is None or len(payload_bits) < expected_bits: payload_bits.append(g & 1)
        if expected_bits is None or len(payload_bits) < expected_bits: payload_bits.append(b & 1)

    if inverted_hint:
        payload_bits = [1 - b for b in payload_bits]

    payload = _bits_to_bytes(payload_bits)
    if payload_len is not None:
        payload = payload[:payload_len]
    info = {"filename": filename, "encrypted": encrypted, "salt": salt, "iters": iters, "length": payload_len}
    return info, payload