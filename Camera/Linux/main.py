#!/usr/bin/env python3
"""
GR2 Camera Linux – entry point.

Usage:
    python main.py                # GUI + web server (default)
    python main.py --no-web       # GUI only, no web server
    python main.py --web-only     # headless web server (no GUI)

Default password: admin123  (change on first login via Settings)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


import config
from app_controller import AppController
from camera_capture import CameraCapture
from gpio_handler import GPIOHandler
from web_server import start_web_server


def main():
    args     = sys.argv[1:]
    web_only = "--web-only" in args
    no_web   = "--no-web"   in args

    # Ensure config.json exists with defaults
    cfg = config.load()
    print(f"[GR2] Config: {config.CONFIG_FILE}")
    print(f"[GR2] Captures dir: {config.get_captures_dir()}")

    # Shared state (GUI ↔ web server)
    ctrl = AppController()

    # GPIO handler (real on Pi, stub on desktop)
    gpio = GPIOHandler()

    # Start camera early so web stream is available immediately
    cam_cfg = cfg["camera"]
    cam = CameraCapture(
        camera_id=cam_cfg["device_id"],
        frame_width=cam_cfg["frame_width"],
        frame_height=cam_cfg["frame_height"],
        fps=cam_cfg["fps"],
    )
    cam.start()
    ctrl.set_frame_getter(cam.get_frame)

    # Start web server in background unless disabled
    if not no_web:
        start_web_server(ctrl, port=cfg["web"]["port"])

    if web_only:
        print("[GR2] Headless mode (--web-only). Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            cam.stop()
        return

    # ---- GUI mode ------------------------------------------------
    import tkinter as tk
    from gui_app import LoginWindow, MainWindow

    root = tk.Tk()

    # Build main window immediately (camera already running — lock screen on top)
    main_win = MainWindow(root, ctrl, gpio, cam)

    def on_login():
        main_win.unlock()

    LoginWindow(root, ctrl, gpio, on_success=on_login)
    root.mainloop()


if __name__ == "__main__":
    main()
