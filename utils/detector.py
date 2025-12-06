"""detector.py - YOLO-based person detector with tracking support"""

from ultralytics import YOLO
import numpy as np


class YOLOPersonDetector:
    """YOLO detector wrapper for person detection with optional tracking."""
    
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.3, use_tracking=True):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.use_tracking = use_tracking

    def detect(self, frame):
        """Detect persons in frame.
        
        Returns:
            list of tuples: (x1, y1, x2, y2, confidence, track_id)
            track_id is None if tracking is disabled
        """
        if self.use_tracking:
            res = self.model.track(frame, persist=True, conf=self.conf_thresh, classes=[0], verbose=False)
        else:
            res = self.model(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        
        if isinstance(res, list):
            res = res[0]
        dets = []
        boxes = res.boxes
        if boxes is None:
            return []
        
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
            # class 0 is person for COCO
            if conf < self.conf_thresh or cls != 0:
                continue
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
            x1, y1, x2, y2 = map(float, xyxy)
            # get track_id if available
            track_id = None
            if self.use_tracking and hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0]) if hasattr(box.id, '__getitem__') else int(box.id)
            dets.append((x1, y1, x2, y2, conf, track_id))
        return dets
