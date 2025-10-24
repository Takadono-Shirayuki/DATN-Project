import cv2
from common import assign_id_for_person, cleanup_ids

def crop_and_resize(frame, center, box_size, resize=(224, 224)):
    x1 = int(center[0] - box_size[0] / 2)
    y1 = int(center[1] - box_size[1] / 2)
    x2 = int(center[0] + box_size[0] / 2)
    y2 = int(center[1] + box_size[1] / 2)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    resized = cv2.resize(cropped, resize)
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
        center_x = int((keypoints[5][0] + keypoints[6][0] + keypoints[11][0] + keypoints[12][0]) / 4)
        center_y = int((keypoints[5][1] + keypoints[6][1] + keypoints[11][1] + keypoints[12][1]) / 4)
        center = (center_x, center_y)
        height = estimate_height(keypoints)
        if height is None:
            continue
        output_frames[id] = {}
        output_frames[id]['image'] = crop_and_resize(input_frame, center, (height, height), resize=(224, 224))
        output_frames[id]['label'] = label
    cleanup_ids(active_labels)
    return output_frames
