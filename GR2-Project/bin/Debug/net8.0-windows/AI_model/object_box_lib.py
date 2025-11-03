import cv2
import numpy as np
from common import assign_id_for_person, cleanup_ids

def crop_and_resize(frame, center, box_size, resize=(224, 224)):
    # box_size is (width, height)
    box_w = int(round(box_size[0]))
    box_h = int(round(box_size[1]))
    x1 = int(round(center[0] - box_w / 2))
    y1 = int(round(center[1] - box_h / 2))
    x2 = x1 + box_w
    y2 = y1 + box_h

    img_h, img_w = frame.shape[:2]

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

    # resize to target
    try:
        resized = cv2.resize(canvas, resize)
    except Exception:
        # fallback: return None if resize fails
        return None
    return resized

def estimate_height(keypoints):
    if keypoints[5][2] < 0.3 or keypoints[6][2] < 0.3 or keypoints[11][2] < 0.3 or keypoints[12][2] < 0.3:
        return None
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    height = 4.5 * abs(hip_y - shoulder_y)
    return height

def separate_object(input_frame, pose_model):
    output_frames = {}
    active_labels = set()
    pose_results = pose_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
    for box, kp in zip(pose_results[0].boxes.xyxy, pose_results[0].keypoints.data):
        keypoints = [[round(x, 2), round(y, 2), round(c, 2)] for x, y, c in kp.tolist()]
        cropped = input_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        id, label, standing = assign_id_for_person(keypoints, cropped)
        if not standing:
            continue
        active_labels.add(id)
        height = estimate_height(keypoints)
        center_x = int((keypoints[5][0] + keypoints[6][0] + keypoints[11][0] + keypoints[12][0]) / 4)
        center_y = int((keypoints[5][1] + keypoints[6][1] + keypoints[11][1] + keypoints[12][1]) / 4) + height // 8
        center = (center_x, center_y)
        if height is None:
            continue
        output_frames[id] = {}
        output_frames[id]['image'] = crop_and_resize(input_frame, center, (height, height), resize=(224, 224))
        output_frames[id]['label'] = label
    cleanup_ids(active_labels)
    return output_frames
