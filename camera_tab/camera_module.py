"""
Camera module for real-time video processing
"""
import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from camera_tab.camera_lib import segmentation
from utils import YOLOPersonDetector
import logging
import os
import sys

# Add parent directory to path for action_classifier_module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from action_classifier_module import ActionRecognizer
    ACTION_CLASSIFIER_AVAILABLE = True
except ImportError:
    ACTION_CLASSIFIER_AVAILABLE = False
    print("⚠️ ActionRecognizer not available. Action recognition disabled.")

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
        
        # Action recognition
        self.action_recognizer = None
        self.enable_action_recognition = False
        self.action_results = {}  # {track_id: {'action': str, 'confidence': float}}
        self.action_thread = None
        self.action_queue = queue.Queue(maxsize=100)  # Queue for async inference
        self.action_thread_running = False
        
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
        elif isinstance(input_source, str):
            # Check if it's a video file
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg')
            if input_source.lower().endswith(video_extensions):
                pass  # Use as is
            else:
                # Assume it's IP address for DroidCam
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
        
        # Clear old action queue and results (in case of restart)
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
        self.action_results.clear()
        
        # Load action recognizer if available (always load if possible, enable/disable via flag)
        if ACTION_CLASSIFIER_AVAILABLE and self.action_recognizer is None:
            try:
                model_path = os.path.join('Behavier_recognition', 'action_classifier_128_2.pth')
                if not os.path.exists(model_path):
                    model_path = 'action_classifier_128_2.pth'  # Try current directory
                
                if os.path.exists(model_path):
                    self.action_recognizer = ActionRecognizer(
                        model_path=model_path,
                        sequence_length=30,
                        min_confidence=0.3,
                        min_valid_keypoints=8
                    )
                    print(f"✅ Action recognizer loaded from: {model_path}")
                    
                    if self.enable_action_recognition:
                        print("✅ Action recognition enabled")
                else:
                    print(f"⚠️ Model file not found: {model_path}")
            except Exception as e:
                import traceback
                print(f"⚠️ Could not load action recognizer: {e}")
                traceback.print_exc()
                self.enable_action_recognition = False
        
        # Start/restart action inference thread if recognizer is loaded
        if self.action_recognizer is not None:
            # Stop old thread if exists
            if self.action_thread_running:
                self.action_thread_running = False
                if self.action_thread:
                    self.action_thread.join(timeout=0.5)
            
            # Start new thread
            self.action_thread_running = True
            self.action_thread = threading.Thread(target=self._action_inference_loop, daemon=True)
            self.action_thread.start()
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        
        # Stop action inference thread
        if self.action_thread_running:
            self.action_thread_running = False
            self.action_queue.put(None)  # Poison pill
            if self.action_thread:
                self.action_thread.join(timeout=1.0)
        
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_active(self):
        """Check if module is actively processing (not ended)"""
        return self.running and self.thread is not None and self.thread.is_alive()
    
    def _action_inference_loop(self):
        """Separate thread for action recognition inference (non-blocking)"""
        try:
            while self.action_thread_running:
                try:
                    # Get keypoints from queue (with timeout to check thread status)
                    item = self.action_queue.get(timeout=0.1)
                    if item is None:  # Poison pill
                        break
                    
                    track_id, keypoints, bbox = item
                    
                    # Update sequence buffer with bbox for coordinate transformation
                    self.action_recognizer.update(track_id, keypoints, bbox)
                    
                    # Get prediction
                    result = self.action_recognizer.predict(track_id, use_smoothing=True)
                    
                    # Store result
                    if result['ready']:
                        self.action_results[track_id] = {
                            'action': result['action'],
                            'confidence': result['confidence']
                        }
                except queue.Empty:
                    continue
                except Exception as e:
                    # Log errors but don't crash the thread
                    print(f"❌ Action inference error: {e}")
        except Exception as e:
            print(f"💥 Action inference thread crashed: {e}")
    
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
                active_track_ids = []
                
                for box, kp in zip(pose_results[0].boxes, pose_results[0].keypoints.data):
                    keypoints = [[round(x, 2), round(y, 2), round(c, 2)] for x, y, c in kp.tolist()]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = [x1, y1, x2, y2]
                    
                    # Get track_id from pose_model tracking
                    track_id = int(box.id[0]) if box.id is not None else None
                    if track_id:
                        active_track_ids.append(track_id)
                    
                    # Send keypoints + bbox to action recognition queue (non-blocking)
                    if self.enable_action_recognition and self.action_recognizer and track_id:
                        try:
                            self.action_queue.put_nowait((track_id, keypoints, bbox))
                        except queue.Full:
                            pass  # Skip if queue full
                    
                    # Determine label (with action if available)
                    base_label = self.custom_ids.get(track_id, f"Person_{track_id}" if track_id else "Person")
                    
                    # Add action to label if available
                    if track_id and track_id in self.action_results:
                        action_info = self.action_results[track_id]
                        action_label = f"{base_label}-{action_info['action']}"
                        conf_text = f"({action_info['confidence']:.2f})"
                    else:
                        action_label = base_label
                        conf_text = ""
                    
                    # Draw bbox if enabled
                    if self.enable_bbox:
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(output_frame, action_label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if conf_text:
                            cv2.putText(output_frame, conf_text, (x1, y2 + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Draw pose skeleton
                    self._draw_pose(output_frame, keypoints)
                
                # Cleanup inactive persons from action recognizer
                if self.enable_action_recognition and self.action_recognizer:
                    self.action_recognizer.cleanup_inactive(active_track_ids)
        
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
        from camera_tab.camera_lib import skeleton
        
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
    
    def set_options(self, bbox=None, pose=None, segmentation=None, action_recognition=None):
        """Update processing options"""
        if bbox is not None:
            self.enable_bbox = bbox
        if pose is not None:
            self.enable_pose = pose
        if segmentation is not None:
            self.enable_segmentation = segmentation
        if action_recognition is not None and ACTION_CLASSIFIER_AVAILABLE:
            old_value = self.enable_action_recognition
            self.enable_action_recognition = action_recognition
            if old_value != action_recognition:
                print(f"{'\u2705' if action_recognition else '\u274c'} Action recognition: {'ENABLED' if action_recognition else 'DISABLED'}")
            # Note: Action recognizer will be initialized on next start() call
    
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
