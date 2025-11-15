import sys
import cv2
import numpy as np

def crop_and_resize(frame, center, box_size, resize=(224, 224), keypoints=None):
    """
    Crop and resize frame with aspect ratio preservation.
    Optionally transform keypoints to match the cropped/resized coordinates.
    
    Args:
        keypoints: list of [x, y, conf] in original frame coordinates (optional)
    
    Returns:
        (resized_frame, transformed_keypoints) if keypoints provided, else resized_frame
    """
    # box_size is (width, height)
    box_w = int(round(box_size[0]))
    box_h = int(round(box_size[1]))
    img_h, img_w = frame.shape[:2]

    # Use provided box_size directly (no quantization). Recompute crop coordinates.
    x1 = int(round(center[0] - box_w / 2))
    y1 = int(round(center[1] - box_h / 2))
    x2 = x1 + box_w
    y2 = y1 + box_h

    # compute intersection with image
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(img_w, x2)
    iy2 = min(img_h, y2)

    # prepare black canvas of the requested box size
    if frame.ndim == 2:
        canvas = np.zeros((box_h, box_w), dtype=frame.dtype)
    else:
        channels = frame.shape[2]
        canvas = np.zeros((box_h, box_w, channels), dtype=frame.dtype)

    # If there's an overlap, copy the overlapping region into the canvas
    if ix1 < ix2 and iy1 < iy2:
        src = frame[iy1:iy2, ix1:ix2]

        # destination start coords on canvas
        dst_x = max(0, -x1)
        dst_y = max(0, -y1)

        h = iy2 - iy1
        w = ix2 - ix1

        canvas[dst_y:dst_y + h, dst_x:dst_x + w] = src

    # If there was no overlap (crop completely outside), canvas remains black

    # Resize while preserving aspect ratio and pad with black to `resize` target.
    target_w, target_h = resize[0], resize[1]
    if box_w <= 0 or box_h <= 0 or target_w <= 0 or target_h <= 0:
        return None

    # compute scale to fit canvas into target while keeping aspect ratio
    scale = min(float(target_w) / float(box_w), float(target_h) / float(box_h))
    new_w = max(1, int(round(box_w * scale)))
    new_h = max(1, int(round(box_h * scale)))

    try:
        resized_small = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return None

    # create final black target canvas
    if frame.ndim == 2:
        final = np.zeros((target_h, target_w), dtype=frame.dtype)
    else:
        channels = frame.shape[2]
        final = np.zeros((target_h, target_w, channels), dtype=frame.dtype)

    # center the resized_small in final
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    final[y_off:y_off + new_h, x_off:x_off + new_w] = resized_small
    
    # Transform keypoints if provided
    if keypoints is not None and len(keypoints) > 0:
        transformed_kp = []
        for x, y, c in keypoints:
            # Step 1: translate to crop coordinates (subtract crop origin)
            kp_x = x - x1
            kp_y = y - y1
            
            # Step 2: scale to resized coordinates
            kp_x = kp_x * scale
            kp_y = kp_y * scale
            
            # Step 3: translate to final canvas offset
            kp_x += x_off
            kp_y += y_off
            
            transformed_kp.append([kp_x, kp_y, c])
        
        return final, transformed_kp
    
    return final

def separate_object(input_frame, detector, seg_model=None, pose_model=None):
    """
    Detect persons using YOLOPersonDetector and crop them.
    Optionally apply binary segmentation (black/white) to isolate each person.
    Optionally detect pose keypoints for each person.
    
    Args:
        input_frame: Input video frame
        detector: YOLOPersonDetector instance
        seg_model: YOLO segmentation model (if None, no segmentation applied)
        pose_model: YOLO pose model (if None, no pose detection)
    
    Returns:
        dict: {person_id: {'image': cropped_frame, 'label': label, 'keypoints': list}}
    """
    output_frames = {}
    active_labels = set()
    # Get detections from detector (returns list of (x1, y1, x2, y2, conf, track_id))
    detections = detector.detect(input_frame)
    
    # Get segmentation masks if seg_model is provided
    masks_data = None
    boxes_data = None
    if seg_model is not None:
        seg_results = seg_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
        if seg_results[0].masks is not None:
            masks_data = seg_results[0].masks.data.cpu().numpy()
            boxes_data = seg_results[0].boxes.xyxy.cpu().numpy()
    
    # Get pose keypoints if pose_model is provided
    pose_keypoints_data = None
    pose_boxes_data = None
    if pose_model is not None:
        pose_results = pose_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
        if pose_results[0].keypoints is not None:
            pose_keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
            pose_boxes_data = pose_results[0].boxes.xyxy.cpu().numpy()
    
    
    for det in detections:
        x1, y1, x2, y2, conf, track_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        bbox_h = y2 - y1
        if bbox_h <= 0:
            continue
        
        # Crop person region
        cropped = input_frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        
        # Use track_id as person ID
        if track_id is not None:
            id = f"Person_{int(track_id)}"
            label = id
        else:
            id = f"Person_Unknown_{len(output_frames)}"
            label = id
        
        active_labels.add(id)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)
        height = bbox_h
        
        # Apply segmentation before cropping to ensure correct alignment
        frame_to_crop = input_frame
        if masks_data is not None and boxes_data is not None:
            # Find best matching mask for this detection bbox
            best_mask_idx = None
            best_iou = 0.0
            
            for idx, seg_box in enumerate(boxes_data):
                # Compute IoU between detection box and segmentation box
                sx1, sy1, sx2, sy2 = seg_box
                ix1 = max(x1, sx1)
                iy1 = max(y1, sy1)
                ix2 = min(x2, sx2)
                iy2 = min(y2, sy2)
                
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (sx2 - sx1) * (sy2 - sy1)
                union_area = box1_area + box2_area - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_mask_idx = idx
            
            # Apply the matched mask to create binary output (white person, black background)
            if best_mask_idx is not None and best_iou > 0.3:  # threshold IoU
                mask = masks_data[best_mask_idx]
                h, w = input_frame.shape[:2]
                mask_resized = cv2.resize(mask, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Create binary frame: white (255) for person, black (0) for background
                frame_to_crop = np.zeros_like(input_frame)
                frame_to_crop[mask_binary == 1] = 255
        # Match pose keypoints with this detection
        keypoints = None
        if pose_keypoints_data is not None and pose_boxes_data is not None:
            best_pose_idx = None
            best_pose_iou = 0.0
            
            for idx, pose_box in enumerate(pose_boxes_data):
                px1, py1, px2, py2 = pose_box
                ix1 = max(x1, px1)
                iy1 = max(y1, py1)
                ix2 = min(x2, px2)
                iy2 = min(y2, py2)
                
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (px2 - px1) * (py2 - py1)
                union_area = box1_area + box2_area - inter_area
                
                pose_iou = inter_area / union_area if union_area > 0 else 0
                
                if pose_iou > best_pose_iou:
                    best_pose_iou = pose_iou
                    best_pose_idx = idx
            
            if best_pose_idx is not None and best_pose_iou > 0.3:
                # Extract keypoints: shape is (num_keypoints, 3) where each is [x, y, confidence]
                kp = pose_keypoints_data[best_pose_idx]
                keypoints = [[float(x), float(y), float(c)] for x, y, c in kp.tolist()]
        
        # Crop and resize from the masked frame (with keypoint transformation)
        if keypoints is not None and len(keypoints) > 0:
            cropped_img, transformed_kp = crop_and_resize(
                frame_to_crop, center, (height, height), resize=(224, 224), keypoints=keypoints
            )
        else:
            cropped_img = crop_and_resize(frame_to_crop, center, (height, height), resize=(224, 224))
            transformed_kp = []
        
        output_frames[id] = {}
        output_frames[id]['image'] = cropped_img
        output_frames[id]['label'] = label
        output_frames[id]['keypoints'] = transformed_kp
    return output_frames
