@echo off
REM ============================================================
REM GR2 Camera Linux — virtual environment setup (Windows)
REM ============================================================
setlocal

set VENV_DIR=venv

echo === GR2 Camera Linux - venv setup ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [Python] Found Python %PY_VER%

REM Create venv
if not exist "%VENV_DIR%\" (
    echo [venv] Creating virtual environment in .\%VENV_DIR% ...
    python -m venv %VENV_DIR%
) else (
    echo [venv] Using existing .\%VENV_DIR%
)

echo [venv] Activating...
call %VENV_DIR%\Scripts\activate.bat

echo [pip] Upgrading pip...
python -m pip install --quiet --upgrade pip

echo [pip] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo === Setup complete ===
echo.
echo To run the app:
echo   %VENV_DIR%\Scripts\activate.bat
echo   python main.py
echo.
echo Options:
echo   python main.py --no-web     (disable web UI)
echo   python main.py --web-only   (headless mode)
echo.
echo Default login password: admin123
echo.
pause
