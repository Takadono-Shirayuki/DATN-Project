"""
GPU-accelerated version using YOLO segmentation tracking (no separate detector needed)
Uses YOLOv8 seg.track() for detection + tracking + segmentation in one model
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop_and_resize_gpu(frame_tensor, center, box_size, resize=(224, 224), keypoints=None):
    """
    GPU-accelerated crop and resize using PyTorch CUDA
    
    Args:
        frame_tensor: torch.Tensor on GPU (C, H, W) format
        center: (x, y) tuple
        box_size: (width, height) tuple
        resize: (width, height) target size
        keypoints: list of [x, y, conf] in frame coordinates
    
    Returns:
        Resized tensor on GPU, transformed keypoints
    """
    box_w, box_h = int(round(box_size[0])), int(round(box_size[1]))
    _, img_h, img_w = frame_tensor.shape
    
    x1 = int(round(center[0] - box_w / 2))
    y1 = int(round(center[1] - box_h / 2))
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    # Compute intersection with image
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(img_w, x2)
    iy2 = min(img_h, y2)
    
    # Create canvas on GPU
    canvas = torch.zeros((frame_tensor.shape[0], box_h, box_w), 
                         dtype=frame_tensor.dtype, device=frame_tensor.device)
    
    # Copy overlapping region
    if ix1 < ix2 and iy1 < iy2:
        src = frame_tensor[:, iy1:iy2, ix1:ix2]
        dst_x = max(0, -x1)
        dst_y = max(0, -y1)
        h = iy2 - iy1
        w = ix2 - ix1
        canvas[:, dst_y:dst_y + h, dst_x:dst_x + w] = src
    
    # Resize with aspect ratio preservation
    target_w, target_h = resize[0], resize[1]
    if box_w <= 0 or box_h <= 0:
        return None, []
    
    scale = min(float(target_w) / float(box_w), float(target_h) / float(box_h))
    new_w = max(1, int(round(box_w * scale)))
    new_h = max(1, int(round(box_h * scale)))
    
    # Resize using PyTorch (GPU accelerated)
    canvas_unsqueezed = canvas.unsqueeze(0)  # Add batch dimension
    resized_small = F.interpolate(canvas_unsqueezed, size=(new_h, new_w), 
                                   mode='bilinear', align_corners=False)
    resized_small = resized_small.squeeze(0)  # Remove batch dimension
    
    # Create final canvas and center the resized content
    final = torch.zeros((frame_tensor.shape[0], target_h, target_w), 
                       dtype=frame_tensor.dtype, device=frame_tensor.device)
    
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    final[:, y_off:y_off + new_h, x_off:x_off + new_w] = resized_small
    
    # Transform keypoints
    if keypoints is not None and len(keypoints) > 0:
        transformed_kp = []
        for x, y, c in keypoints:
            kp_x = (x - x1) * scale + x_off
            kp_y = (y - y1) * scale + y_off
            transformed_kp.append([kp_x, kp_y, c])
        return final, transformed_kp
    
    return final, []


def batch_crop_and_resize_gpu_tracking(frame, detections, seg_results, pose_results=None, enable_segmentation=True):
    """
    Batch process all detections using GPU operations
    Optimized for seg model with tracking (detections extracted from seg_results)
    
    Args:
        frame: numpy array (H, W, C) - can be scaled or original frame
        detections: list of (x1, y1, x2, y2, conf, track_id) - extracted from seg_results
        seg_results: segmentation results from YOLO with tracking
        pose_results: pose results from YOLO (optional)
        enable_segmentation: whether to apply segmentation mask (default: True)
    
    Returns:
        dict: {person_id: {'image': numpy array, 'label': label, 'keypoints': list}}
    """
    if len(detections) == 0:
        return {}
    
    # Convert frame to tensor on GPU (C, H, W) format
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda()
    
    output_frames = {}
    
    # Extract masks from seg_results (already has tracking)
    masks_data = None
    if seg_results is not None and seg_results[0].masks is not None:
        masks_data = seg_results[0].masks.data  # Already on GPU
    
    # Extract pose data
    pose_keypoints_data = None
    pose_boxes_data = None
    if pose_results is not None and pose_results[0].keypoints is not None:
        pose_keypoints_data = pose_results[0].keypoints.data  # Already on GPU
        pose_boxes_data = pose_results[0].boxes.xyxy  # Already on GPU
    
    # IoU matching for pose (seg already matched via track_id)
    pose_matches = {}
    
    if pose_keypoints_data is not None and pose_boxes_data is not None:
        det_boxes = torch.tensor([[d[0], d[1], d[2], d[3]] for d in detections], device='cuda')
        
        # Vectorized IoU computation
        for det_idx in range(len(detections)):
            det_box = det_boxes[det_idx]
            
            ix1 = torch.maximum(det_box[0], pose_boxes_data[:, 0])
            iy1 = torch.maximum(det_box[1], pose_boxes_data[:, 1])
            ix2 = torch.minimum(det_box[2], pose_boxes_data[:, 2])
            iy2 = torch.minimum(det_box[3], pose_boxes_data[:, 3])
            
            inter_area = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
            box1_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
            box2_area = (pose_boxes_data[:, 2] - pose_boxes_data[:, 0]) * (pose_boxes_data[:, 3] - pose_boxes_data[:, 1])
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / (union_area + 1e-7)
            
            # Check if iou tensor is not empty before argmax
            if iou.numel() > 0:
                best_idx = torch.argmax(iou).item()
                best_iou = iou[best_idx].item()
                
                if best_iou > 0.3:
                    pose_matches[det_idx] = best_idx
    
    # Process each detection
    for det_idx, det in enumerate(detections):
        x1, y1, x2, y2, conf, track_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            continue
        
        # Generate ID from track_id
        person_id = f"Person_{int(track_id)}" if track_id is not None else f"Person_Unknown_{det_idx}"
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)
        height = max(bbox_h, bbox_w)  # Use the longer dimension
        
        # Apply segmentation mask (only if enabled)
        frame_to_crop = frame_tensor
        if enable_segmentation and masks_data is not None and det_idx < len(masks_data):
            mask = masks_data[det_idx]  # Direct indexing - seg model maintains order
            
            # Resize mask to frame size on GPU
            mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                        size=(frame_tensor.shape[1], frame_tensor.shape[2]),
                                        mode='bilinear', align_corners=False)
            mask_binary = (mask_resized > 0.5).squeeze()
            
            # Apply mask on GPU
            frame_to_crop = torch.zeros_like(frame_tensor)
            frame_to_crop[:, mask_binary] = 255
        
        # Get keypoints (match with pose via IoU)
        keypoints = None
        if det_idx in pose_matches:
            pose_idx = pose_matches[det_idx]
            kp = pose_keypoints_data[pose_idx].cpu().numpy()
            keypoints = [[float(x), float(y), float(c)] for x, y, c in kp.tolist()]
        
        # Crop and resize on GPU
        cropped_tensor, transformed_kp = crop_and_resize_gpu(
            frame_to_crop, center, (height, height), resize=(224, 224), keypoints=keypoints
        )
        
        if cropped_tensor is None:
            continue
        
        # Convert back to numpy (C, H, W) -> (H, W, C)
        cropped_img = cropped_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        output_frames[person_id] = {
            'image': cropped_img,
            'label': person_id,
            'keypoints': transformed_kp
        }
    
    return output_frames


def separate_object_gpu_tracking(input_frame, seg_model, pose_model=None, 
                                  scale_factor=1.0, tracker='bytetrack.yaml',
                                  enable_segmentation=True):
    """
    GPU-accelerated version using segmentation model with built-in tracking
    Eliminates need for separate detection model
    
    Args:
        input_frame: Input video frame (numpy array)
        seg_model: YOLO segmentation model (uses .track() method)
        pose_model: YOLO pose model (optional)
        scale_factor: Resolution scale factor for inference (0.0-1.0).
                     1.0 = full resolution (slower, best quality)
                     0.5 = half resolution (faster)
                     0.25 = quarter resolution (fastest)
        tracker: Tracker config ('bytetrack.yaml' or 'botsort.yaml')
        enable_segmentation: whether to apply segmentation mask (default: True)
    
    Returns:
        dict: {person_id: {'image': cropped_frame, 'label': label, 'keypoints': list}}
    """
    # Resize frame for faster inference if needed
    if scale_factor < 1.0:
        orig_h, orig_w = input_frame.shape[:2]
        target_w = int(orig_w * scale_factor)
        target_h = int(orig_h * scale_factor)
        input_frame = cv2.resize(input_frame, (target_w, target_h), 
                                interpolation=cv2.INTER_LINEAR)
    
    # Use seg model with tracking - replaces separate detector!
    seg_results = seg_model.track(
        source=input_frame,
        persist=True,           # Persist tracks across frames
        tracker=tracker,        # ByteTrack or BoT-SORT
        conf=0.5,
        verbose=False,
        classes=[0]             # Only person class (class 0 in COCO)
    )
    
    # Check if any detections
    if seg_results[0].boxes is None or len(seg_results[0].boxes) == 0:
        return {}
    
    # Extract detections from seg_results (with track_ids!)
    boxes = seg_results[0].boxes
    detections = []
    
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = boxes.conf[i].item()
        
        # Get track_id (may be None for first frame or lost tracks)
        track_id = boxes.id[i].item() if boxes.id is not None else None
        
        detections.append((
            float(xyxy[0]), float(xyxy[1]), 
            float(xyxy[2]), float(xyxy[3]), 
            conf, track_id
        ))
    
    # Run pose estimation on same frame
    pose_results = None
    if pose_model is not None:
        pose_results = pose_model.predict(
            source=input_frame, 
            save=False, 
            conf=0.5, 
            verbose=False
        )
    
    # Batch process all detections on GPU
    return batch_crop_and_resize_gpu_tracking(input_frame, detections, seg_results, pose_results, enable_segmentation)
