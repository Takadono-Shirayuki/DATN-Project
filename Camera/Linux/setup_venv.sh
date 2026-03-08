#!/usr/bin/env bash
# ============================================================
# GR2 Camera Linux — virtual environment setup (Linux / macOS)
# ============================================================
set -e

VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"

echo "=== GR2 Camera Linux — venv setup ==="
echo ""

# Verify Python
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] Python3 not found. Install python3 and try again."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print('%d.%d' % sys.version_info[:2])")
echo "[Python] Found $PYTHON ($PY_VER)"

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[venv] Creating virtual environment in ./$VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[venv] Using existing ./$VENV_DIR"
fi

echo "[venv] Activating..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[pip] Upgrading pip..."
pip install --quiet --upgrade pip

echo "[pip] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Options:"
echo "  python main.py --no-web    # disable web UI"
echo "  python main.py --web-only  # headless mode"
echo ""
echo "Default login password: admin123"
