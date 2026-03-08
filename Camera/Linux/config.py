"""
Runtime configuration management.

config.json is auto-created on first run with secure defaults.
Default password: admin123  (change immediately after first login)
"""

import copy
import hashlib
import json
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"
PBKDF2_ITERATIONS = 260_000
DEFAULT_PASSWORD = "admin123"

_DEFAULTS: dict = {
    "server":  {"ip": "192.168.1.100", "port": 3001},
    "camera":  {"device_id": 0, "frame_width": 640, "frame_height": 480, "fps": 30},
    "auth":    {"password_salt": "", "password_hash": ""},
    "device":  {"name": "gr2-camera-linux", "type": "recognition"},
    "storage": {"captures_dir": ""},
    "web":     {"port": 8080, "secret_key": ""},
}


def _create_default() -> dict:
    """Generate config.json with a hashed default password and random secrets."""
    cfg = copy.deepcopy(_DEFAULTS)
    salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac("sha256", DEFAULT_PASSWORD.encode(), salt, PBKDF2_ITERATIONS)
    cfg["auth"]["password_salt"] = salt.hex()
    cfg["auth"]["password_hash"] = dk.hex()
    cfg["storage"]["captures_dir"] = str(Path.home() / "gr2_captures")
    cfg["web"]["secret_key"] = os.urandom(32).hex()
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def load() -> dict:
    if not CONFIG_FILE.exists():
        return _create_default()
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ---- convenience helpers -----------------------------------------------

def get_server() -> tuple:
    """Return (ip: str, port: int)."""
    cfg = load()
    return cfg["server"]["ip"], int(cfg["server"]["port"])


def set_server(ip: str, port: int) -> None:
    cfg = load()
    cfg["server"]["ip"] = ip.strip()
    cfg["server"]["port"] = int(port)
    save(cfg)


def get_captures_dir() -> Path:
    """Return captures root directory, creating sub-dirs if needed."""
    cfg = load()
    d = Path(cfg["storage"]["captures_dir"])
    (d / "photos").mkdir(parents=True, exist_ok=True)
    (d / "videos").mkdir(parents=True, exist_ok=True)
    return d
