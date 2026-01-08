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
from utils.text_utils import put_text_with_background
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

try:
    from gait_recognizer_module import GaitRecognizer
    GAIT_RECOGNIZER_AVAILABLE = True
except ImportError:
    GAIT_RECOGNIZER_AVAILABLE = False
    print("⚠️ GaitRecognizer not available. Gait recognition disabled.")

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
        
        # Gait recognition
        self.gait_recognizer = None
        self.enable_gait_recognition = True  # Bật mặc định
        self.gait_labels = {}  # {track_id: 'Person Name'}
        
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
        
        # Load gait recognizer if available
        if GAIT_RECOGNIZER_AVAILABLE and self.gait_recognizer is None and self.enable_gait_recognition:
            try:
                self.gait_recognizer = GaitRecognizer(
                    model_path='open_set/encoder_resnet.pth',
                    database_path='database.json',  # OpenSetGaitMatcher format
                    buffer_size=30,  # 30 frames
                    cooldown_frames=60  # 2 seconds cooldown
                )
                self.gait_labels.clear()
                print("✅ Gait recognizer loaded with OpenSetGaitMatcher")
            except Exception as e:
                print(f"⚠️ Could not load gait recognizer: {e}")
                self.enable_gait_recognition = False
        
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
                    try:
                        self.action_recognizer.update(track_id, keypoints, bbox)
                    except Exception as e:
                        print(f"❌ ActionRecognizer.update error for {track_id}: {e}")
                        raise

                    # Get prediction
                    try:
                        result = self.action_recognizer.predict(track_id, use_smoothing=True)
                    except Exception as e:
                        print(f"❌ ActionRecognizer.predict error for {track_id}: {e}")
                        raise

                    # Store result and debug-log
                    try:
                        if isinstance(result, dict):
                            if result.get('ready'):
                                self.action_results[track_id] = {
                                    'action': result.get('action'),
                                    'confidence': result.get('confidence')
                                }
                        else:
                            pass
                    except Exception:
                        pass
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
        
        # Run segmentation model để có masks cho gait recognition
        seg_results = None
        if self.enable_gait_recognition and self.gait_recognizer and self.seg_model:
            seg_results = self.seg_model.track(source=frame, persist=True, save=False, conf=0.5, verbose=False, classes=[0])
        
        if self.enable_segmentation:
            output_frame = segmentation(frame, output_frame, self.seg_model, type='Foreground')
        
        if self.enable_pose:
            # When pose enabled, use pose_model (it has bbox already)
            pose_results = self.pose_model.track(source=frame, persist=True, save=False, conf=0.5, verbose=False)
            
            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                active_track_ids = []
                
                for box_idx, (box, kp) in enumerate(zip(pose_results[0].boxes, pose_results[0].keypoints.data)):
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
                    
                    # Gait recognition: update when person is walking
                    gait_label = None
                    is_walking = False
                    if self.enable_gait_recognition and self.gait_recognizer and track_id:
                        # Check if person is walking
                        if track_id in self.action_results:
                            action_info = self.action_results[track_id]
                            is_walking = action_info['action'] == 'Walking'
                        
                        # Extract person mask/silhouette for GEI từ seg_results
                        person_mask = self._extract_person_mask_from_segresults(frame, bbox, box_idx, seg_results)
                        
                        # Update gait recognizer with mask
                        recognized_name = self.gait_recognizer.update(
                            person_id=f"Person_{track_id}",
                            frame=None,  # Không dùng frame gốc nữa
                            is_walking=is_walking,
                            mask=person_mask,
                            keypoints=keypoints
                        )
                        
                        if recognized_name:
                            self.gait_labels[track_id] = recognized_name
                            gait_label = recognized_name
                        elif track_id in self.gait_labels:
                            gait_label = self.gait_labels[track_id]
                    
                    # Determine label (priority: gait > custom > track_id)
                    if gait_label:
                        base_label = gait_label
                    else:
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
                        # Use Vietnamese-compatible text rendering
                        output_frame = put_text_with_background(
                            output_frame, action_label, (x1, y1 - 30),
                            font_size=16, text_color=(0, 255, 0), 
                            bg_color=(0, 0, 0), padding=3
                        )
                        if conf_text:
                            output_frame = put_text_with_background(
                                output_frame, conf_text, (x1, y2 + 5),
                                font_size=12, text_color=(0, 255, 0),
                                bg_color=(0, 0, 0), padding=2
                            )
                    
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
                # Use Vietnamese-compatible text rendering
                output_frame = put_text_with_background(
                    output_frame, label, (int(x1), int(y1) - 30),
                    font_size=16, text_color=(0, 255, 0),
                    bg_color=(0, 0, 0), padding=3
                )
        
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
    
    def _extract_person_mask_from_segresults(self, frame, bbox, box_idx, seg_results):
        """
        Extract binary mask/silhouette từ seg_results (giống object_box_lib)
        Crop and resize with aspect ratio preservation
        
        Args:
            frame: Original frame (H, W, 3)
            bbox: [x1, y1, x2, y2] bounding box của người
            box_idx: Index của detection trong pose results để match với seg_results
            seg_results: Results từ seg_model.track()
        
        Returns:
            Binary mask (224, 224) - grayscale silhouette với padding, hoặc None nếu không có mask
        """
        if seg_results is None or seg_results[0].masks is None:
            return None
        
        try:
            masks_data = seg_results[0].masks.data.cpu().numpy()
            
            # Match mask với bbox dựa trên IoU
            x1, y1, x2, y2 = bbox
            best_mask = None
            best_iou = 0.0
            
            if seg_results[0].boxes is not None:
                seg_boxes = seg_results[0].boxes
                
                for mask_idx, (mask, seg_box) in enumerate(zip(masks_data, seg_boxes)):
                    # Check class
                    cls = int(seg_box.cls[0]) if hasattr(seg_box.cls, '__getitem__') else int(seg_box.cls)
                    if cls != 0:  # Only person
                        continue
                    
                    # Compute IoU
                    seg_xyxy = seg_box.xyxy[0].cpu().numpy()
                    sx1, sy1, sx2, sy2 = seg_xyxy
                    
                    ix1 = max(x1, sx1)
                    iy1 = max(y1, sy1)
                    ix2 = min(x2, sx2)
                    iy2 = min(y2, sy2)
                    
                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    box1_area = (x2 - x1) * (y2 - y1)
                    box2_area = (sx2 - sx1) * (sy2 - sy1)
                    union_area = box1_area + box2_area - inter_area
                    
                    iou = inter_area / (union_area + 1e-7)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = mask
            
            if best_mask is None or best_iou < 0.3:
                return None
            
            # Resize mask to frame size
            img_h, img_w = frame.shape[:2]
            mask_resized = cv2.resize(best_mask, (img_w, img_h))
            mask_binary = ((mask_resized > 0.5) * 255).astype(np.uint8)
            
            # Apply mask to frame (chỉ giữ phần trong mask)
            masked_frame = np.zeros_like(frame)
            masked_frame[mask_binary > 127] = 255  # White silhouette
            
            # Tính center và max_side từ bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            max_side = max(bbox_w, bbox_h)
            
            # Tạo square crop region
            crop_x1 = int(center_x - max_side / 2)
            crop_y1 = int(center_y - max_side / 2)
            crop_x2 = crop_x1 + int(max_side)
            crop_y2 = crop_y1 + int(max_side)
            
            # Tính intersection với frame
            src_x1 = max(0, crop_x1)
            src_y1 = max(0, crop_y1)
            src_x2 = min(img_w, crop_x2)
            src_y2 = min(img_h, crop_y2)
            
            dst_x1 = max(0, -crop_x1)
            dst_y1 = max(0, -crop_y1)
            
            # Tạo canvas vuông với padding
            canvas = np.zeros((int(max_side), int(max_side)), dtype=np.uint8)
            
            # Copy masked region vào canvas (convert to grayscale)
            if src_x1 < src_x2 and src_y1 < src_y2:
                src_region = masked_frame[src_y1:src_y2, src_x1:src_x2]
                if len(src_region.shape) == 3:
                    src_region = cv2.cvtColor(src_region, cv2.COLOR_BGR2GRAY)
                h = src_y2 - src_y1
                w = src_x2 - src_x1
                canvas[dst_y1:dst_y1 + h, dst_x1:dst_x1 + w] = src_region
            
            # Resize to 224x224 với aspect ratio preservation
            canvas_h, canvas_w = canvas.shape[:2]
            scale = min(224.0 / canvas_w, 224.0 / canvas_h)
            new_w = max(1, int(canvas_w * scale))
            new_h = max(1, int(canvas_h * scale))
            
            # Resize mask
            mask_small = cv2.resize(canvas, (new_w, new_h))
            
            # Center trong 224x224 canvas
            final_mask = np.zeros((224, 224), dtype=np.uint8)
            x_off = (224 - new_w) // 2
            y_off = (224 - new_h) // 2
            final_mask[y_off:y_off + new_h, x_off:x_off + new_w] = mask_small
            
            return final_mask
            
        except Exception as e:
            print(f"Error extracting person mask from seg_results: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_person_mask(self, frame, bbox):
        """
        Extract binary mask/silhouette for a person using segmentation model
        Crop and resize with aspect ratio preservation (giống crop_and_resize_gpu)
        
        Args:
            frame: Original frame (H, W, 3)
            bbox: [x1, y1, x2, y2] bounding box của người
        
        Returns:
            Binary mask (224, 224) - grayscale silhouette với padding, hoặc None nếu không có mask
        """
        if self.seg_model is None:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            img_h, img_w = frame.shape[:2]
            
            # Tính center và box_size (giống crop_and_resize_gpu)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            max_side = max(bbox_w, bbox_h)
            
            # Tạo square crop region
            crop_x1 = int(center_x - max_side / 2)
            crop_y1 = int(center_y - max_side / 2)
            crop_x2 = crop_x1 + int(max_side)
            crop_y2 = crop_y1 + int(max_side)
            
            # Tính intersection với frame
            src_x1 = max(0, crop_x1)
            src_y1 = max(0, crop_y1)
            src_x2 = min(img_w, crop_x2)
            src_y2 = min(img_h, crop_y2)
            
            dst_x1 = max(0, -crop_x1)
            dst_y1 = max(0, -crop_y1)
            
            # Tạo canvas vuông với padding
            canvas = np.zeros((int(max_side), int(max_side), 3), dtype=np.uint8)
            
            # Copy region vào canvas
            if src_x1 < src_x2 and src_y1 < src_y2:
                src_region = frame[src_y1:src_y2, src_x1:src_x2]
                h = src_y2 - src_y1
                w = src_x2 - src_x1
                canvas[dst_y1:dst_y1 + h, dst_x1:dst_x1 + w] = src_region
            
            # Run segmentation trên canvas
            seg_results = self.seg_model.predict(source=canvas, save=False, conf=0.5, verbose=False)
            
            if seg_results[0].masks is not None and seg_results[0].boxes is not None:
                masks = seg_results[0].masks.data.cpu().numpy()
                boxes = seg_results[0].boxes
                
                # Tìm mask của người (class 0)
                canvas_h, canvas_w = canvas.shape[:2]
                for mask, box in zip(masks, boxes):
                    cls = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
                    if cls == 0:  # Person class
                        # Resize mask về kích thước canvas
                        mask_resized = cv2.resize(mask, (canvas_w, canvas_h))
                        # Convert sang binary [0, 255]
                        binary_mask = ((mask_resized > 0.5) * 255).astype(np.uint8)
                        
                        # Resize to 224x224 với aspect ratio preservation
                        scale = min(224.0 / canvas_w, 224.0 / canvas_h)
                        new_w = max(1, int(canvas_w * scale))
                        new_h = max(1, int(canvas_h * scale))
                        
                        # Resize mask
                        mask_small = cv2.resize(binary_mask, (new_w, new_h))
                        
                        # Center trong 224x224 canvas
                        final_mask = np.zeros((224, 224), dtype=np.uint8)
                        x_off = (224 - new_w) // 2
                        y_off = (224 - new_h) // 2
                        final_mask[y_off:y_off + new_h, x_off:x_off + new_w] = mask_small
                        
                        return final_mask
            
            return None
            
        except Exception as e:
            print(f"Error extracting person mask: {e}")
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
