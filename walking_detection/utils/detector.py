"""detector.py - YOLO-based person detector with GPU support"""

import torch
from ultralytics import YOLO
import numpy as np

class YOLOPersonDetector:
    """YOLO detector wrapper for person detection with optional tracking."""
    
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.3, use_tracking=True):
        # 1. Automatic Device Detection (MPS for Mac, CUDA for Nvidia)
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        print(f"--- YOLO running on: {self.device.upper()} ---")

        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.use_tracking = use_tracking

    def detect(self, frame):
        """Detect persons in frame."""
        # Pass the 'device' argument to YOLO
        if self.use_tracking:
            res = self.model.track(frame, persist=True, conf=self.conf_thresh, classes=[0], verbose=False, device=self.device)
        else:
            res = self.model(frame, conf=self.conf_thresh, classes=[0], verbose=False, device=self.device)
        
        if isinstance(res, list):
            res = res[0]
            
        dets = []
        if res.boxes is None:
            return []
        
        for box in res.boxes:
            # Safe float conversion
            conf = float(box.conf[0]) if box.conf.numel() > 0 else 0.0
            cls = int(box.cls[0]) if box.cls.numel() > 0 else -1
            
            # class 0 is person
            if conf < self.conf_thresh or cls != 0:
                continue
                
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            
            # Get track_id safely
            track_id = None
            if self.use_tracking and box.id is not None:
                track_id = int(box.id[0])
                
            dets.append((x1, y1, x2, y2, conf, track_id))
            
        return dets