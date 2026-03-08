"""
Thread-safe state bridge between the GUI (main thread) and the web server
(background daemon thread).

The GUI writes state via update(); the web server reads it via get_state().
The web server queues commands via enqueue(); the GUI polls via drain().
"""

import queue
import threading

MODE_PHOTO  = "photo"
MODE_VIDEO  = "video"
MODE_STREAM = "stream"
MODES = (MODE_PHOTO, MODE_VIDEO, MODE_STREAM)


class AppController:
    def __init__(self):
        self._lock = threading.Lock()
        self._mode          = MODE_STREAM
        self._recording     = False
        self._streaming     = False
        self._stream_failed  = False
        self._stream_fail_addr = ""
        self._status         = "Starting..."
        self._fps           = 0.0
        self._cmd: queue.Queue = queue.Queue()

    # ---- state (readable from any thread) ---------------------------

    def get_state(self) -> dict:
        with self._lock:
            return {
                "mode":          self._mode,
                "recording":     self._recording,
                "streaming":     self._streaming,
                "stream_failed":     self._stream_failed,
                "stream_fail_addr":  self._stream_fail_addr,
                "status":            self._status,
                "fps":           round(self._fps, 1),
            }

    def update(self, **kwargs) -> None:
        """Called by the GUI thread to publish new state."""
        with self._lock:
            for key, val in kwargs.items():
                setattr(self, f"_{key}", val)

    # ---- command queue (web server -> GUI) ---------------------------

    def enqueue(self, cmd: str, arg=None) -> None:
        """Web server pushes a command; GUI processes it on next poll."""
        self._cmd.put((cmd, arg))

    def drain(self):
        """Yield all pending (cmd, arg) pairs without blocking."""
        while True:
            try:
                yield self._cmd.get_nowait()
            except queue.Empty:
                return

    # ---- frame getter (GUI -> web server) ----------------------------

    def set_frame_getter(self, fn) -> None:
        """Register a callable that returns the latest camera frame (or None)."""
        self._frame_getter = fn

    def get_frame(self):
        """Return the latest camera frame, or None if unavailable."""
        fn = getattr(self, "_frame_getter", None)
        return fn() if fn else None
