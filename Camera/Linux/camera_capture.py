# Camera Capture Module
# handles camera frame acquisition

import cv2
import threading
from collections import deque

class CameraCapture:
    def __init__(self, camera_id=0, frame_width=640, frame_height=480, fps=30):
        """
        Initialize camera capture
        
        Args:
            camera_id: camera device ID (0 = default)
            frame_width: frame width
            frame_height: frame height
            fps: frames per second
        """
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.frame_buffer = deque(maxlen=2)  # Keep last 2 frames
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start camera capture thread"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop camera capture thread"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
    
    def _capture_loop(self):
        """Internal capture loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
    
    def get_frame(self):
        """
        Get latest frame from buffer
        
        Returns:
            frame: latest captured frame or None
        """
        if self.frame_buffer:
            return self.frame_buffer[-1]
        return None
    
    def is_available(self):
        """Check if camera is available"""
        return self.cap.isOpened()
