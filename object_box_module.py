"""
Object Box module for person detection and extraction
"""
import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from detector import YOLOPersonDetector
from recorder import Recorder
from object_box_lib import separate_object
from pose_visualizer import draw_keypoints_on_frame
import logging

class ObjectBoxModule:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Models
        self.detector = None
        self.seg_model = None
        self.pose_model = None
        self.recorder = None
        
        # Settings
        self.enable_pose = False
        self.enable_segmentation = False
        self.tracking_id = None
        self.current_person_index = 0  # For << >> navigation
        self.person_list = []  # List of detected person IDs
        
        # Custom ID mapping: {track_id: custom_id}
        self.custom_ids = {}
        
        # Suppress YOLO logs
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    def start(self, input_source):
        """
        Start object box processing
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
        
        # Load models
        if self.detector is None:
            self.detector = YOLOPersonDetector(model_path='yolov8n.pt', 
                                              conf_thresh=0.3, use_tracking=True)
        if self.seg_model is None:
            self.seg_model = YOLO('yolov8s-seg.pt')
        if self.pose_model is None:
            self.pose_model = YOLO('yolov8s-pose.pt')
        
        # Initialize recorder
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.recorder is None:
            self.recorder = Recorder(out_dir='person_videos', fps=fps, 
                                    frame_size=(224, 224), timeout=5.0)
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        """Stop object box processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Close recorder
        if self.recorder:
            try:
                self.recorder.close_all()
            except:
                pass
    
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
            output_frame, metadata = self._process_frame(frame)
            
            # Put in queue (non-blocking, drop old frames if full)
            try:
                self.frame_queue.put_nowait((output_frame, metadata))
            except queue.Full:
                # Remove old frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((output_frame, metadata))
                except:
                    pass
    
    def _process_frame(self, frame):
        """Process a single frame based on enabled features"""
        # Separate objects (persons)
        output_frames = separate_object(
            frame, 
            self.detector, 
            seg_model=self.seg_model if self.enable_segmentation else None,
            pose_model=self.pose_model if self.enable_pose else None,
        )
        
        # Update person list
        self.person_list = list(output_frames.keys())
        
        # Select which person to display
        if self.tracking_id not in output_frames:
            if len(output_frames) == 0:
                output_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                selected_id = None
            else:
                # Get first person
                selected_id = self.person_list[0] if self.person_list else None
                output_frame = output_frames[selected_id]['image'] if selected_id else np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            selected_id = self.tracking_id
            output_frame = output_frames[selected_id]['image']
        
        # Draw skeleton if enabled
        if self.enable_pose and selected_id is not None and selected_id in output_frames:
            keypoints = output_frames[selected_id].get('keypoints')
            if keypoints is not None and len(keypoints) > 0:
                output_frame = draw_keypoints_on_frame(output_frame, keypoints)
        
        # Record all persons
        try:
            for pid, data in output_frames.items():
                img = data.get('image')
                keypoints = data.get('keypoints')
                if img is not None:
                    self.recorder.update(pid, img, keypoints=keypoints)
        except Exception:
            pass
        
        # Metadata with custom IDs
        display_id = self.custom_ids.get(selected_id, selected_id) if selected_id else None
        display_list = [self.custom_ids.get(pid, pid) for pid in self.person_list]
        
        metadata = {
            'selected_id': display_id,
            'person_count': len(output_frames),
            'person_list': display_list,
            'original_selected_id': selected_id  # Keep original for internal tracking
        }
        
        return output_frame, metadata
    
    def get_frame(self, timeout=0.1):
        """
        Get the latest processed frame
        Returns: (numpy array, metadata dict) or (None, None) if no frame available
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def set_options(self, pose=None, segmentation=None):
        """Update processing options"""
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
            object_box.assign_custom_id(1, '123')  # Person 1 will now show as '123'
        """
        self.custom_ids[track_id] = custom_id
    
    def remove_custom_id(self, track_id):
        """Remove custom ID assignment
        
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
    
    def assign_custom_id(self, track_id, custom_id):
        """Assign a custom ID to a tracked person
        
        Args:
            track_id: The original track_id from YOLO (e.g., 1, 2, 3)
            custom_id: Your custom ID string (e.g., '123', 'John', 'Employee_A')
        
        Example:
            object_box.assign_custom_id(1, '123')  # Person 1 will now show as '123'
            object_box.assign_custom_id(2, 'John') # Person 2 will now show as 'John'
        """
        self.custom_ids[track_id] = custom_id
    
    def remove_custom_id(self, track_id):
        """Remove custom ID assignment, revert to default track_id
        
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
    
    def navigate_person(self, direction):
        """
        Navigate between detected persons
        direction: 'prev' or 'next'
        """
        if not self.person_list:
            return
        
        if direction == 'prev':
            self.current_person_index = (self.current_person_index - 1) % len(self.person_list)
        elif direction == 'next':
            self.current_person_index = (self.current_person_index + 1) % len(self.person_list)
        
        if self.person_list:
            self.tracking_id = self.person_list[self.current_person_index]
