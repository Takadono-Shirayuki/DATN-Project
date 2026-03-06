# YOLOv8 Detector Module
# for camera app - performs object detection and pose estimation

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', pose_model_path='yolov8s-pose.pt', seg_model_path='yolov8n-seg.pt'):
        """
        Initialize YOLO detectors
        
        Args:
            model_path: path to object detection model
            pose_model_path: path to pose estimation model
        """
        self.detector = YOLO(model_path)
        self.pose_model = YOLO(pose_model_path)
        self.seg_model = None

        # Segmentation model is optional. If available, it creates cleaner silhouettes.
        seg_path = Path(seg_model_path)
        if seg_path.exists():
            self.seg_model = YOLO(str(seg_path))
    
    def detect_persons(self, frame):
        """
        Detect persons in frame
        
        Args:
            frame: input image
            
        Returns:
            detections: list of detection boxes
        """
        results = self.detector(frame, classes=[0])  # class 0 = person
        return results
    
    def estimate_pose(self, frame):
        """
        Estimate pose (skeleton) of persons
        
        Args:
            frame: input image
            
        Returns:
            pose_results: list of pose keypoints
        """
        results = self.pose_model(frame)
        return results
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: input image
            results: YOLO detection results
            
        Returns:
            annotated_frame: frame with annotations
        """
        annotated_frame = results[0].plot()
        return annotated_frame

    def extract_person_silhouette(self, frame, detections):
        """
        Extract binary silhouette (white person on black background).

        Priority:
        1) Use segmentation masks if seg model exists.
        2) Fallback to person boxes from detector.

        Returns:
            silhouette_bgr: 3-channel silhouette image for preview/sending
            person_count: number of detected persons
        """
        h, w = frame.shape[:2]
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        # Path 1: segmentation-based silhouette
        if self.seg_model is not None:
            seg_results = self.seg_model(frame, classes=[0], verbose=False)
            if seg_results and seg_results[0].masks is not None:
                masks = seg_results[0].masks.data.cpu().numpy()
                for mask in masks:
                    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_mask[resized > 0.5] = 255

                silhouette_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                return silhouette_bgr, len(masks)

        # Path 2: detection-box fallback silhouette
        person_count = 0
        if detections and detections[0].boxes is not None:
            boxes = detections[0].boxes.xyxy.cpu().numpy()
            person_count = len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Rough foreground extraction inside person box.
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, roi_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_mask[y1:y2, x1:x2] = np.maximum(binary_mask[y1:y2, x1:x2], roi_mask)

        silhouette_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        return silhouette_bgr, person_count
