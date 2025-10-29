# crypto/pbkdf2_aesgcm.py
from __future__ import annotations
import os
from typing import Tuple

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEFAULT_ITERS = 200_000
KEY_LEN = 32  # AES-256

def derive_key(password: str, salt: bytes, iters: int = DEFAULT_ITERS) -> bytes:
    """Derive an AES-256 key from a password using PBKDF2-HMAC(SHA256)."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=salt,
        iterations=iters,
    )
    return kdf.derive(password.encode("utf-8"))

def encrypt_aes_gcm(plaintext: bytes, password: str, iters: int = DEFAULT_ITERS, aad: bytes | None = None) -> tuple[bytes, bytes, int]:
    """
    Returns (ciphertext_with_nonce, salt, iterations).
    We prefix the ciphertext with the 12B nonce.
    """
    salt = os.urandom(16)
    key = derive_key(password, salt, iters)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data=aad)
    return nonce + ct, salt, iters

def decrypt_aes_gcm(cipher_with_nonce: bytes, password: str, salt: bytes, iters: int, aad: bytes | None = None) -> bytes:
    nonce = cipher_with_nonce[:12]
    ct = cipher_with_nonce[12:]
    key = derive_key(password, salt, iters)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, associated_data=aad)