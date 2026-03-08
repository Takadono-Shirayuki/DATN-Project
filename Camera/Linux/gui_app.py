"""
GR2 Camera Linux – Tkinter GUI Application.

Three capture modes
  photo   — save single JPEG frames
  video   — record .avi video files
  stream  — stream frames to the recognition server via WebSocket

Authentication
  At startup a login dialog is shown.
  Physical button (GPIO17, active-low) bypasses the login dialog and
  goes straight to the main window.  On non-GPIO platforms use F1.

Cross-platform: tested on Windows (venv) and Linux VM; deployable to
embedded hardware via buildroot.
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

import auth
import config
from app_controller import AppController, MODE_PHOTO, MODE_VIDEO, MODE_STREAM
from camera_capture import CameraCapture
from gpio_handler import GPIOHandler
from network_client import NetworkClient

PREVIEW_W = 640
PREVIEW_H = 480


# ===========================================================================
# Login window
# ===========================================================================

class LoginWindow:
    """
    Modal login dialog shown at startup.

    - Password verification uses PBKDF2-SHA256.
    - GPIO17 (or F1 on desktop) bypasses the dialog entirely.
    """

    def __init__(self, root: tk.Tk, ctrl: AppController,
                 gpio: GPIOHandler, on_success):
        self._root = root
        self._ctrl = ctrl
        self._gpio = gpio
        self._on_success = on_success

        self.win = tk.Toplevel(root)
        self.win.title("GR2 Camera – Login")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
        self._build()
        self.win.grab_set()
        self.win.focus()

        # GPIO / F1 bypass
        self._gpio._callback = self._gpio_bypass
        self.win.bind("<F1>", lambda _e: self._gpio_bypass())

    # ---- UI --------------------------------------------------------

    def _build(self):
        frm = ttk.Frame(self.win, padding=28)
        frm.pack()

        ttk.Label(frm, text="GR2 Camera System",
                  font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))

        ttk.Label(frm, text="Password:").pack(anchor="w")
        self._pw_var = tk.StringVar()
        self._entry = ttk.Entry(frm, textvariable=self._pw_var,
                                show="*", width=28)
        self._entry.pack(pady=(4, 6))
        self._entry.bind("<Return>", lambda _: self._try_login())

        self._err_lbl = ttk.Label(frm, text="", foreground="red")
        self._err_lbl.pack(pady=(0, 8))

        ttk.Button(frm, text="Login", command=self._try_login,
                   width=18).pack()

        hint = ("Physical button (GPIO17) bypasses login."
                if self._gpio.available else
                "Press F1 to simulate the physical button.")
        ttk.Label(frm, text=hint, foreground="gray",
                  font=("Segoe UI", 8)).pack(pady=(14, 0))

        self._entry.focus()

    # ---- actions ---------------------------------------------------

    def _try_login(self):
        if auth.verify(self._pw_var.get()):
            self._complete()
        else:
            self._err_lbl.config(text="Incorrect password")
            self._pw_var.set("")
            self._entry.focus()

    def _gpio_bypass(self):
        """Called from GPIO interrupt or F1 key -- route to main thread."""
        self._root.after(0, self._complete)

    def _complete(self):
        self.win.grab_release()
        self.win.destroy()
        self._on_success()


# ===========================================================================
# Captures viewer
# ===========================================================================

class ViewerWindow:
    """Browse and play back captured photos and videos."""

    def __init__(self, master: tk.Tk):
        self.win = tk.Toplevel(master)
        self.win.title("GR2 – Captures Viewer")
        self.win.geometry("960x560")
        self._playing = False
        self._tk_img = None
        self._build()
        self._refresh()

    # ---- UI --------------------------------------------------------

    def _build(self):
        pane = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: file list
        left = ttk.Frame(pane, width=240)
        pane.add(left, weight=1)

        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True)

        ph_frm = ttk.Frame(nb)
        vd_frm = ttk.Frame(nb)
        nb.add(ph_frm, text="Photos")
        nb.add(vd_frm, text="Videos")

        self._photo_lb = tk.Listbox(ph_frm, selectmode=tk.BROWSE)
        self._photo_lb.pack(fill=tk.BOTH, expand=True)
        self._photo_lb.bind("<<ListboxSelect>>", self._on_photo_select)

        self._video_lb = tk.Listbox(vd_frm, selectmode=tk.BROWSE)
        self._video_lb.pack(fill=tk.BOTH, expand=True)
        self._video_lb.bind("<<ListboxSelect>>", self._on_video_select)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Refresh",
                   command=self._refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Open Folder",
                   command=self._open_folder).pack(side=tk.LEFT, padx=2)

        # Right: preview canvas
        right = ttk.Frame(pane)
        pane.add(right, weight=3)

        self._canvas = tk.Canvas(right, bg="black",
                                 width=PREVIEW_W, height=PREVIEW_H)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        self._info_lbl = ttk.Label(right, text="Select a file to preview",
                                   wraplength=700)
        self._info_lbl.pack(pady=4)

    # ---- data refresh ----------------------------------------------

    def _refresh(self):
        d = config.get_captures_dir()
        self._photo_lb.delete(0, tk.END)
        self._video_lb.delete(0, tk.END)
        for p in sorted((d / "photos").glob("*.jpg"), reverse=True):
            self._photo_lb.insert(tk.END, p.name)
        for v in sorted((d / "videos").glob("*.avi"), reverse=True):
            self._video_lb.insert(tk.END, v.name)

    # ---- preview ---------------------------------------------------

    def _on_photo_select(self, _event):
        sel = self._photo_lb.curselection()
        if not sel:
            return
        path = config.get_captures_dir() / "photos" / self._photo_lb.get(sel[0])
        try:
            img = Image.open(path)
            img.thumbnail((PREVIEW_W, PREVIEW_H))
            self._tk_img = ImageTk.PhotoImage(img)
            self._canvas.config(width=img.width, height=img.height)
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
            self._info_lbl.config(text=str(path))
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.win)

    def _on_video_select(self, _event):
        sel = self._video_lb.curselection()
        if not sel:
            return
        self._playing = False          # stop any current playback
        time.sleep(0.05)
        path = config.get_captures_dir() / "videos" / self._video_lb.get(sel[0])
        self._info_lbl.config(text=f"Video: {path}")
        self._playing = True
        threading.Thread(target=self._play_video,
                         args=(path,), daemon=True).start()

    def _play_video(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / fps
        while cap.isOpened() and self._playing:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((PREVIEW_W, PREVIEW_H))
            tk_img = ImageTk.PhotoImage(img)
            # Schedule GUI update on main thread
            self._canvas.after(0, self._show_video_frame,
                               tk_img, img.width, img.height)
            time.sleep(delay)
        cap.release()

    def _show_video_frame(self, tk_img, w: int, h: int):
        self._tk_img = tk_img   # keep reference to prevent GC
        self._canvas.config(width=w, height=h)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _open_folder(self):
        import platform
        import subprocess
        d = str(config.get_captures_dir())
        sys_name = platform.system()
        if sys_name == "Windows":
            os.startfile(d)
        elif sys_name == "Darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])


# ===========================================================================
# Settings dialog
# ===========================================================================

class SettingsDialog:
    """Change server IP/port and account password."""

    def __init__(self, master: tk.Tk):
        self.win = tk.Toplevel(master)
        self.win.title("Settings")
        self.win.resizable(False, False)
        self.win.grab_set()
        self._build()

    def _build(self):
        nb = ttk.Notebook(self.win)
        nb.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Server tab
        sf = ttk.Frame(nb, padding=16)
        nb.add(sf, text="Server")

        ip, port = config.get_server()
        ttk.Label(sf, text="Server IP:").grid(row=0, column=0,
                                              sticky="w", pady=5)
        self._ip_var = tk.StringVar(value=ip)
        ttk.Entry(sf, textvariable=self._ip_var,
                  width=22).grid(row=0, column=1, padx=8)

        ttk.Label(sf, text="Port:").grid(row=1, column=0,
                                         sticky="w", pady=5)
        self._port_var = tk.StringVar(value=str(port))
        ttk.Entry(sf, textvariable=self._port_var,
                  width=8).grid(row=1, column=1, sticky="w", padx=8)

        self._srv_msg = ttk.Label(sf, text="")
        self._srv_msg.grid(row=2, column=0, columnspan=2, pady=4)
        ttk.Button(sf, text="Save",
                   command=self._save_server).grid(row=3, column=0,
                                                   columnspan=2)

        # Password tab
        pf = ttk.Frame(nb, padding=16)
        nb.add(pf, text="Password")

        rows = [("Current password:", "old"),
                ("New password:",     "new"),
                ("Confirm new:",      "confirm")]
        self._pw_vars: dict = {}
        for i, (lbl, key) in enumerate(rows):
            ttk.Label(pf, text=lbl).grid(row=i, column=0,
                                          sticky="w", pady=5)
            v = tk.StringVar()
            self._pw_vars[key] = v
            ttk.Entry(pf, textvariable=v, show="*",
                      width=22).grid(row=i, column=1, padx=8)

        self._pw_msg = ttk.Label(pf, text="")
        self._pw_msg.grid(row=3, column=0, columnspan=2, pady=4)
        ttk.Button(pf, text="Change Password",
                   command=self._change_pw).grid(row=4, column=0,
                                                  columnspan=2)

        ttk.Button(self.win, text="Close",
                   command=self.win.destroy).pack(pady=8)

    def _save_server(self):
        import ipaddress
        ip = self._ip_var.get().strip()
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            self._srv_msg.config(text="Invalid IP address",
                                 foreground="red")
            return
        try:
            port = int(self._port_var.get())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            self._srv_msg.config(text="Invalid port number",
                                 foreground="red")
            return
        config.set_server(ip, port)
        self._srv_msg.config(text="Saved successfully",
                             foreground="green")

    def _change_pw(self):
        old     = self._pw_vars["old"].get()
        new     = self._pw_vars["new"].get()
        confirm = self._pw_vars["confirm"].get()
        if new != confirm:
            self._pw_msg.config(text="New passwords do not match",
                                foreground="red")
            return
        if len(new) < 4:
            self._pw_msg.config(text="Password too short (min 4 chars)",
                                foreground="red")
            return
        if auth.change_password(old, new):
            self._pw_msg.config(text="Password changed successfully",
                                foreground="green")
            for v in self._pw_vars.values():
                v.set("")
        else:
            self._pw_msg.config(text="Incorrect current password",
                                foreground="red")


# ===========================================================================
# Main application window
# ===========================================================================

class MainWindow:
    """
    Main camera control window shown after successful authentication.

    Layout
    ┌──────────────┬──────────────────────────────────────┐
    │  [Photo]     │                                      │
    │  [Video]     │      Camera preview  (640×480)       │
    │  [Stream]    │                                      │
    │  ─────────── ├──────────────────────────────────────┤
    │  [Captures]  │   [ Action button ]                  │
    │  [Settings]  │                                      │
    └──────────────┴──────────────────────────────────────┘
      Status bar
    """

    def __init__(self, root: tk.Tk, ctrl: AppController,
                 gpio: GPIOHandler, cam: CameraCapture):
        self.root = root
        self._ctrl = ctrl
        self._gpio = gpio

        # Camera (started before login, passed in)
        self._cam = cam
        ctrl.set_frame_getter(self._cam.get_frame)

        # Internal state
        self._locked        = True    # locked until login is completed
        self._mode          = MODE_STREAM
        self._recording     = False
        self._streaming     = False
        self._cam_paused    = False   # True while stream mode is active
        self._canvas_msg    = ""      # text drawn on canvas when paused
        self._canvas_msg_color = "white"
        self._countdown_id  = None    # pending after() id for stream countdown
        self._video_writer: cv2.VideoWriter = None
        self._net_client:   NetworkClient   = None
        self._tk_img        = None          # prevent GC of current frame

        # FPS tracking
        self._frame_count = 0
        self._fps_time    = time.monotonic()

        self.root.title("GR2 Camera System")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

        # Re-assign GPIO callback to main-window action
        self._gpio._callback = self._gpio_callback

        # Start loops
        self._update_preview()
        self._poll_commands()

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        self.root.resizable(False, False)

        # Layout: [left sidebar] | [preview canvas] | [right thumb button]
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- left sidebar -----------------------------------------------
        _SB_W   = 90       # sidebar width in px
        _BTN_BG = "#2c2c2e"
        _BTN_FG = "#e5e5ea"
        _BTN_AB = "#3a3a3c"
        _BTN_FONT_ICON = ("Segoe UI Emoji", 22)
        _BTN_FONT_LBL  = ("Segoe UI", 8, "bold")

        sb = tk.Frame(main, width=_SB_W, bg="#1c1c1e", relief=tk.RIDGE, bd=2)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        def _sb_btn(parent, icon, label, cmd, active_bg=_BTN_AB):
            """Create a sidebar icon-over-text button."""
            frm = tk.Frame(parent, bg=_BTN_BG, cursor="hand2")
            frm.pack(fill=tk.X, padx=4, pady=3)
            tk.Label(frm, text=icon, font=_BTN_FONT_ICON,
                     bg=_BTN_BG, fg=_BTN_FG).pack(pady=(6, 0))
            tk.Label(frm, text=label, font=_BTN_FONT_LBL,
                     bg=_BTN_BG, fg=_BTN_FG).pack(pady=(0, 6))
            # bind click on both labels and the frame itself
            for w in (frm,) + tuple(frm.winfo_children()):
                w.bind("<Button-1>",  lambda _e, c=cmd: c())
                w.bind("<Enter>",     lambda _e, f=frm: f.config(bg=active_bg))
                w.bind("<Leave>",     lambda _e, f=frm: f.config(bg=_BTN_BG))
            return frm

        self._mode_btns: dict = {}
        _mode_defs = [
            (MODE_PHOTO,  "\U0001F4F7", "Photo"),
            (MODE_VIDEO,  "\U0001F3AC", "Video"),
            (MODE_STREAM, "\U0001F4E1", "Stream"),
        ]
        for mode, icon, label in _mode_defs:
            frm = _sb_btn(sb, icon, label,
                          cmd=lambda m=mode: self._switch_mode(m))
            self._mode_btns[mode] = frm

        tk.Frame(sb, height=2, bg="#48484a").pack(fill=tk.X, padx=8, pady=6)

        _sb_btn(sb, "\U0001F5BC", "Gallery",
                cmd=lambda: ViewerWindow(self.root))

        tk.Frame(sb, height=2, bg="#48484a").pack(fill=tk.X, padx=8, pady=6)

        gpio_txt   = "GPIO17\nactive" if self._gpio.available else "GPIO\nN/A (F1)"
        gpio_color = "#32d74b" if self._gpio.available else "#636366"
        tk.Label(sb, text=gpio_txt, fg=gpio_color, bg="#1c1c1e",
                 font=("Segoe UI", 7), wraplength=80,
                 justify="center").pack(pady=4)

        web_port = config.load().get("web", {}).get("port", 8080)
        tk.Label(sb, text=f"Web UI\n:{web_port}", fg="#636366", bg="#1c1c1e",
                 font=("Segoe UI", 7), justify="center").pack(pady=2)

        # ---- center: preview canvas -------------------------------------
        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(center, width=PREVIEW_W,
                                  height=PREVIEW_H, bg="black")
        self._canvas.pack(padx=6, pady=6)

        # ---- right sidebar: large action button (right-thumb position) --
        # Width ~108px; button sits at bottom so the right thumb reaches it
        # naturally when holding the device in landscape.
        rsb = tk.Frame(main, width=108, bg="#1c1c1e", relief=tk.RIDGE, bd=2)
        rsb.pack(side=tk.LEFT, fill=tk.Y)
        rsb.pack_propagate(False)

        self._action_btn = tk.Button(
            rsb,
            text="\U0001F4F8\nCapture",
            command=self._on_action,
            font=("Segoe UI", 13, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#388E3C",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            pady=32,
            cursor="hand2",
        )
        # pack at bottom so right thumb reaches it when holding the device
        self._action_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=16)

        # Status bar
        self._status_var = tk.StringVar(value="  Ready")
        ttk.Label(self.root, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor="w",
                  padding=(8, 2)).pack(side=tk.BOTTOM, fill=tk.X)

        # F1 simulates GPIO press
        self.root.bind("<F1>", lambda _e: self._gpio_callback())

        self._switch_mode(MODE_STREAM)
        # Start locked: disable action button until login completes
        self._action_btn.config(state=tk.DISABLED)

    # ---------------------------------------------------------------- modes

    def _switch_mode(self, mode: str):
        # Cancel any running countdown
        if self._countdown_id is not None:
            self.root.after_cancel(self._countdown_id)
            self._countdown_id = None
        # Stop current activities
        if self._recording:
            self._stop_recording()
        if self._streaming:
            self._stop_streaming()

        # Always resume camera preview when switching modes
        self._cam_paused = False
        self._canvas_msg = ""

        self._mode = mode
        self._ctrl.update(mode=mode, stream_failed=False)

        # Highlight active mode button
        _ACTIVE_BG  = "#0a84ff"
        _INACTIVE_BG = "#2c2c2e"
        for m, frm in self._mode_btns.items():
            bg = _ACTIVE_BG if m == mode else _INACTIVE_BG
            frm.config(bg=bg)
            for child in frm.winfo_children():
                child.config(bg=bg)

        # Update action button: icon + colour per mode
        _props = {
            MODE_PHOTO:  ("\U0001F4F8\nCapture", "#4CAF50", "#388E3C"),  # green
            MODE_VIDEO:  ("\u23FA\nRecord",      "#607D8B", "#455A64"),  # slate
            MODE_STREAM: ("\u25B6\nStream",       "#2196F3", "#1565C0"),  # blue
        }
        txt, bg, abg  = _props[mode]
        self._action_btn.config(text=txt, bg=bg, activebackground=abg)

        if mode == MODE_STREAM:
            if self._locked:
                # Camera preview runs but don't attempt network connection yet
                self._set_status("Stream mode (locked)")
            else:
                ip, port = config.get_server()
                self._cam_paused = True
                self._set_canvas_message(
                    f"\U0001F4E1  Connecting...\n\n{ip}:{port}", "#93c5fd")
                self._set_status("Connecting to server...")
                threading.Thread(target=self._try_connect,
                                 daemon=True).start()
        else:
            self._set_status(f"Mode: {mode.capitalize()}")

    def _try_connect(self):
        ip, port = config.get_server()
        client = NetworkClient(ip, port)
        ok = client.connect()
        self.root.after(0, self._on_connect_result, ok, client, ip, port)

    def _on_connect_result(self, ok: bool, client,
                           ip: str, port: int):
        if ok:
            self._net_client = client
            # Resume camera preview once connected
            self._cam_paused = False
            self._canvas_msg = ""
            self._ctrl.update(stream_failed=False)
            self._set_status(f"Connected to {ip}:{port} — ready to stream")
        else:
            self._ctrl.update(stream_failed=True,
                              stream_fail_addr=f"{ip}:{port}")
            self._set_status(f"Cannot connect to {ip}:{port}")
            self._start_countdown(5, ip, port)

    def _start_countdown(self, n: int, ip: str, port):
        if n > 0:
            self._cam_paused = True
            self._set_canvas_message(
                f"\u26A0  Connection failed\n\n{ip}:{port}\n\n"
                f"Switching to Photo mode in {n}s...",
                "#fca5a5")
            self._countdown_id = self.root.after(
                1000, self._start_countdown, n - 1, ip, port)
        else:
            self._countdown_id = None
            self._switch_mode(MODE_PHOTO)

    # ---------------------------------------------------------------- actions

    def unlock(self):
        """Called after successful login — enables all controls."""
        self._locked = False
        self._action_btn.config(state=tk.NORMAL)
        # If already in stream mode, kick off the connection now
        if self._mode == MODE_STREAM:
            self._switch_mode(MODE_STREAM)

    def _on_action(self):
        if self._mode == MODE_PHOTO:
            self._capture_photo()
        elif self._mode == MODE_VIDEO:
            self._stop_recording() if self._recording else self._start_recording()
        elif self._mode == MODE_STREAM:
            self._stop_streaming() if self._streaming else self._start_streaming()

    def _capture_photo(self):
        frame = self._cam.get_frame()
        if frame is None:
            self._set_status("No frame available")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        path = config.get_captures_dir() / "photos" / f"photo_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        msg = f"Saved: {path.name}"
        self._set_status(msg)
        self._ctrl.update(status=msg)

    def _start_recording(self):
        frame = self._cam.get_frame()
        if frame is None:
            self._set_status("Camera not ready")
            return
        h, w  = frame.shape[:2]
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = config.get_captures_dir() / "videos" / f"video_{ts}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps_val = config.load()["camera"]["fps"]
        self._video_writer = cv2.VideoWriter(
            str(path), fourcc, fps_val, (w, h)
        )
        self._recording = True
        self._action_btn.config(text="\u23F9\nStop",
                               bg="#F44336", activebackground="#C62828")
        msg = f"Recording -> {path.name}"
        self._set_status(msg)
        self._ctrl.update(recording=True, status=msg)

    def _stop_recording(self):
        self._recording = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self._action_btn.config(text="\u23FA\nRecord",
                               bg="#607D8B", activebackground="#455A64")
        self._set_status("Recording saved")
        self._ctrl.update(recording=False, status="Recording saved")

    def _start_streaming(self):
        if not (self._net_client and self._net_client.is_connected):
            self._set_status("Not connected -- reconnecting...")
            threading.Thread(target=self._try_connect,
                             daemon=True).start()
            return
        cfg_d = config.load()
        self._net_client.send_registration(
            cfg_d["device"]["name"], cfg_d["device"]["type"]
        )
        self._streaming = True
        self._action_btn.config(text="\u23F9\nStop",
                               bg="#FF5722", activebackground="#D84315")
        msg = f"Streaming to {self._net_client.url}"
        self._set_status(msg)
        self._ctrl.update(streaming=True, status=msg)

    def _stop_streaming(self):
        self._streaming = False
        if self._net_client:
            self._net_client.disconnect()
            self._net_client = None
        self._cam_paused = False
        self._canvas_msg = ""
        self._action_btn.config(text="\u25B6\nStream",
                               bg="#2196F3", activebackground="#1565C0")
        self._set_status("Streaming stopped")
        self._ctrl.update(streaming=False, status="Streaming stopped")

    # ---------------------------------------------------------------- preview loop

    def _set_canvas_message(self, text: str, color: str = "white"):
        self._canvas_msg = text
        self._canvas_msg_color = color

    def _update_preview(self):
        if self._cam_paused:
            # Camera is off — draw message overlay on black canvas
            self._canvas.delete("all")
            if self._canvas_msg:
                self._canvas.create_text(
                    PREVIEW_W // 2, PREVIEW_H // 2,
                    text=self._canvas_msg,
                    fill=self._canvas_msg_color,
                    font=("Segoe UI", 17, "bold"),
                    justify="center",
                )
            self.root.after(100, self._update_preview)
            return

        frame = self._cam.get_frame()
        if frame is not None:
            # Write to video file
            if self._recording and self._video_writer:
                self._video_writer.write(frame)

            # Send to server
            if (self._streaming
                    and self._net_client
                    and self._net_client.is_connected):
                self._net_client.send_frame(frame)

            # FPS counter
            self._frame_count += 1
            now = time.monotonic()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                fps = self._frame_count / elapsed
                self._ctrl.update(fps=fps)
                self._frame_count = 0
                self._fps_time = now

            # Render preview
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((PREVIEW_W, PREVIEW_H), Image.NEAREST)
            self._tk_img = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw",
                                      image=self._tk_img)

            # Status overlay
            if self._recording:
                self._canvas.create_oval(10, 10, 28, 28,
                                         fill="red", outline="")
                self._canvas.create_text(34, 19, text="REC",
                                         fill="white",
                                         font=("Arial", 9, "bold"),
                                         anchor="w")
            elif self._streaming:
                self._canvas.create_oval(10, 10, 28, 28,
                                         fill="#00e676", outline="")
                self._canvas.create_text(34, 19, text="LIVE",
                                         fill="white",
                                         font=("Arial", 9, "bold"),
                                         anchor="w")

        self.root.after(33, self._update_preview)   # ~30 fps

    # ---------------------------------------------------------------- command poll

    def _poll_commands(self):
        """Process commands queued by the web server."""
        for cmd, arg in self._ctrl.drain():
            if cmd == "switch_mode" and arg in (
                    MODE_PHOTO, MODE_VIDEO, MODE_STREAM):
                self._switch_mode(arg)
            elif cmd == "action":
                self._on_action()
        self.root.after(250, self._poll_commands)

    # ---------------------------------------------------------------- helpers

    def _gpio_callback(self):
        """Invoked from GPIO interrupt thread -- marshal to main thread."""
        self.root.after(0, self._on_action)

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _on_close(self):
        self._recording = False
        self._streaming = False
        if self._video_writer:
            self._video_writer.release()
        if self._net_client:
            self._net_client.disconnect()
        self._cam.stop()
        self._gpio.cleanup()
        self.root.destroy()
