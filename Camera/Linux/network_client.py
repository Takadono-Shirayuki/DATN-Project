# Network Client Module
# Communicates with the UI server via WebSocket (/camera endpoint)

import json
import threading
import base64
import cv2
import numpy as np
from typing import Optional, Callable

try:
    import websocket  # websocket-client package
except ImportError:
    raise ImportError("websocket-client is required: pip install websocket-client")


class NetworkClient:
    def __init__(self, server_ip: str, server_port: int = 3000, timeout: int = 10):
        """
        Args:
            server_ip:   server IP address
            server_port: server port (default 3000 — same as Vite UI)
            timeout:     seconds to wait for initial connection
        """
        self.url = f"ws://{server_ip}:{server_port}/camera"
        self.timeout = timeout

        self._ws_app: Optional[websocket.WebSocketApp] = None
        self._ws = None           # set in on_open
        self.is_connected = False
        self._thread: Optional[threading.Thread] = None

        self.receive_callback: Optional[Callable] = None

    # ------------------------------------------------------------------ public

    def connect(self) -> bool:
        """Connect to server.  Returns True if connection established."""
        connected_event = threading.Event()

        def on_open(ws):
            self._ws = ws
            self.is_connected = True
            print(f"[NetworkClient] connected to {self.url}")
            connected_event.set()

        def on_message(ws, data):
            if self.receive_callback:
                try:
                    self.receive_callback(json.loads(data))
                except Exception:
                    pass

        def on_error(ws, error):
            print(f"[NetworkClient] error: {error}")
            self.is_connected = False
            connected_event.set()  # unblock connect() on failure

        def on_close(ws, code, msg):
            print(f"[NetworkClient] disconnected (code={code})")
            self.is_connected = False

        self._ws_app = websocket.WebSocketApp(
            self.url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self._thread = threading.Thread(
            target=self._ws_app.run_forever,
            kwargs={'ping_interval': 20, 'ping_timeout': 10},
            daemon=True,
        )
        self._thread.start()

        connected_event.wait(self.timeout)
        return self.is_connected

    def disconnect(self):
        self.is_connected = False
        if self._ws_app:
            self._ws_app.close()

    def send_frame(self, frame: np.ndarray, metadata: dict = None):
        """Encode frame as JPEG and send to server."""
        if not self.is_connected or self._ws is None:
            return
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            message = {
                'type': 'frame',
                'frame': frame_b64,
                'metadata': metadata or {}
            }
            self._ws.send(json.dumps(message))
        except Exception as e:
            print(f"[NetworkClient] send_frame error: {e}")
            self.is_connected = False

    def send_registration(self, device_name: str, device_type: str = 'camera'):
        """Register this device with the server."""
        if not self.is_connected or self._ws is None:
            return
        try:
            message = {
                'type': 'register',
                'device_name': device_name,
                'device_type': device_type,
            }
            self._ws.send(json.dumps(message))
        except Exception as e:
            print(f"[NetworkClient] send_registration error: {e}")
