"""
Camera module for real-time video processing
"""
import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from camera_lib import segmentation
from detector import YOLOPersonDetector
import logging

class CameraModule:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Models
        self.detector = None
        self.pose_model = None
        self.seg_model = None
        
        # Settings
        self.enable_bbox = False
        self.enable_pose = False
        self.enable_segmentation = False
        
        # Custom ID mapping: {track_id: custom_id}
        self.custom_ids = {}
        
        # Suppress YOLO logs
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    def start(self, input_source):
        """
        Start camera capture
        input_source: 0 for webcam, or URL/path for video
        """
        if self.running:
            return False
        
        # Remove quotes if present
        if isinstance(input_source, str):
            input_source = input_source.strip('"').strip("'")
        
        # Parse input
        if input_source == '0' or input_source == 0:
            input_source = 0
        elif isinstance(input_source, str) and input_source.lower().endswith('.mp4'):
            pass  # Use as is
        elif isinstance(input_source, str):
            input_source = f'http://{input_source}:4747/video'
        
        self.cap = cv2.VideoCapture(input_source)
        if not self.cap.isOpened():
            return False
        
        # Load models lazily
        if self.detector is None:
            self.detector = YOLOPersonDetector(model_path='yolov8n.pt', conf_thresh=0.5, use_tracking=True)
        if self.pose_model is None:
            self.pose_model = YOLO('yolov8s-pose.pt')
        if self.seg_model is None:
            self.seg_model = YOLO('yolov8n-seg.pt')
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_active(self):
        """Check if module is actively processing (not ended)"""
        return self.running and self.thread is not None and self.thread.is_alive()
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.running = False  # Mark as stopped when video ends
                break
            
            # Process frame
            output_frame = self._process_frame(frame)
            
            # Put in queue (non-blocking, drop old frames if full)
            try:
                self.frame_queue.put_nowait(output_frame)
            except queue.Full:
                # Remove old frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(output_frame)
                except:
                    pass
    
    def _process_frame(self, frame):
        """Process a single frame based on enabled features"""
        output_frame = frame.copy()
        
        if self.enable_segmentation:
            output_frame = segmentation(frame, output_frame, self.seg_model, type='Foreground')
        
        if self.enable_pose:
            # When pose enabled, use pose_model (it has bbox already)
            pose_results = self.pose_model.track(source=frame, persist=True, save=False, conf=0.5, verbose=False)
            
            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                for box, kp in zip(pose_results[0].boxes, pose_results[0].keypoints.data):
                    keypoints = [[round(x, 2), round(y, 2), round(c, 2)] for x, y, c in kp.tolist()]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Get track_id from pose_model tracking
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    # Draw bbox if enabled
                    if self.enable_bbox:
                        # Use custom_id if assigned, otherwise use track_id
                        if track_id and track_id in self.custom_ids:
                            label = self.custom_ids[track_id]
                        else:
                            label = f"Person_{track_id}" if track_id else "Person"
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw pose skeleton
                    self._draw_pose(output_frame, keypoints)
        
        elif self.enable_bbox:
            # When only bbox enabled (no pose), use lightweight detector
            detections = self.detector.detect(frame)
            
            for det in detections:
                x1, y1, x2, y2, conf, track_id = det
                # Use custom_id if assigned, otherwise use track_id
                if track_id and track_id in self.custom_ids:
                    label = self.custom_ids[track_id]
                else:
                    label = f"Person_{track_id}" if track_id else "Person"
                cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(output_frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_frame
    
    def _draw_pose(self, frame, keypoints):
        """Draw pose skeleton on frame"""
        from camera_lib import skeleton
        
        drawed_keypoints = set()
        for i, j in skeleton:
            x1, y1, c1 = keypoints[i]
            x2, y2, c2 = keypoints[j]
            if c1 > 0.3 and c2 > 0.3:
                if i not in drawed_keypoints or j not in drawed_keypoints:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                for idx, (x, y, c) in zip([i, j], [(x1, y1, c1), (x2, y2, c2)]):
                    if idx not in drawed_keypoints:
                        drawed_keypoints.add(idx)
                        cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 0), -1)
    
    def get_frame(self, timeout=0.1):
        """
        Get the latest processed frame
        Returns: numpy array (BGR image) or None if no frame available
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_options(self, bbox=None, pose=None, segmentation=None):
        """Update processing options"""
        if bbox is not None:
            self.enable_bbox = bbox
        if pose is not None:
            self.enable_pose = pose
        if segmentation is not None:
            self.enable_segmentation = segmentation
    
    def assign_custom_id(self, track_id, custom_id):
        """Assign a custom ID to a tracked person
        
        Args:
            track_id: The original track_id from YOLO (e.g., 1, 2, 3)
            custom_id: Your custom ID string (e.g., '123', 'John', 'Employee_A')
        
        Example:
            camera.assign_custom_id(1, '123')  # Person_1 will now show as '123'
            camera.assign_custom_id(2, 'John') # Person_2 will now show as 'John'
        """
        self.custom_ids[track_id] = custom_id
    
    def remove_custom_id(self, track_id):
        """Remove custom ID assignment, revert to default Person_X
        
        Args:
            track_id: The track_id to remove custom ID from
        """
        if track_id in self.custom_ids:
            del self.custom_ids[track_id]
    
    def clear_custom_ids(self):
        """Clear all custom ID assignments"""
        self.custom_ids.clear()
    
    def get_custom_ids(self):
        """Get current custom ID mappings
        
        Returns:
            dict: {track_id: custom_id}
        """
        return self.custom_ids.copy()
