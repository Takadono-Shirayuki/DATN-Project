"""
Single-account password authentication using PBKDF2-SHA256.

Only one account exists; no registration endpoint is exposed.
"""

import hashlib
import hmac
import os

import config

_ITER = config.PBKDF2_ITERATIONS


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITER).hex()


def verify(password: str) -> bool:
    """Return True if *password* matches the stored hash."""
    cfg = config.load()
    salt = bytes.fromhex(cfg["auth"]["password_salt"])
    expected = cfg["auth"]["password_hash"]
    # constant-time comparison to prevent timing attacks
    return hmac.compare_digest(_hash(password, salt), expected)


def change_password(old: str, new: str) -> bool:
    """
    Change password from *old* to *new*.
    Returns False if *old* is incorrect or *new* is too short.
    """
    if not verify(old):
        return False
    if len(new) < 4:
        return False
    salt = os.urandom(32)
    cfg = config.load()
    cfg["auth"]["password_salt"] = salt.hex()
    cfg["auth"]["password_hash"] = _hash(new, salt)
    config.save(cfg)
    return True
