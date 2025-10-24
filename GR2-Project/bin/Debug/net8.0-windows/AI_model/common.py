import json
import os
import sys
import numpy as np
import cv2
import torch
import clip
from PIL import Image

# Trạng thái điều khiển
class State:
    class Camera:
        enable_bbox = False
        enable_pose = False
        enable_segmentation = False
    class ObjectBox:
        tracking_id = None
        enable_segmentation = False

# Cấu trúc khung xương COCO
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Model CLIP để phân loại tư thế
# Tải mô hình CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"
if os.path.exists("ViT-B-32.pt"):
    clip_model, preprocess = clip.load("./ViT-B-32.pt", device=device)
else:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Danh sách mô tả tư thế
text_labels = [
    "a person walking",
    "a person running",
    "a person not standing",
]
text_tokens = clip.tokenize(text_labels).to(device)

def classify_pose_with_clip(image_np):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_preprocessed = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_preprocessed)
        text_features = clip_model.encode_text(text_tokens)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits[0].cpu().numpy()
    best_index = probs.argmax()
    return text_labels[best_index], probs[best_index]

# Lệnh từ C#
def listen_commands():
    global State
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "bbox_on": State.Camera.enable_bbox = True
        elif cmd == "bbox_off": State.Camera.enable_bbox = False
        elif cmd == "pose_on": State.Camera.enable_pose = True
        elif cmd == "pose_off": State.Camera.enable_pose = False
        elif cmd == "seg_on": State.Camera.enable_segmentation = True; State.ObjectBox.enable_segmentation = True
        elif cmd == "seg_off": State.Camera.enable_segmentation = False; State.ObjectBox.enable_segmentation = False
        elif cmd == "cancel":
            print("Process canceled by command.")
            sys.exit(0)

def write_output(meta, data):
    meta_json = json.dumps(meta).encode('utf-8')
    try:
        sys.stdout.buffer.write(b'--META--\n')
        sys.stdout.buffer.write(meta_json)
        sys.stdout.buffer.write(b'\n--ENDMETA--\n')
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.write(b'\n')
        sys.stdout.flush()
        return True
    except BrokenPipeError:
        print("⚠️ Mất kết nối với tiến trình chính, dừng gửi ảnh.")
        return False

# Gán ID
ids = []
nonce = 0
def assign_id_for_person(keypoints, cropped, conf_threshold=0.3):
    global ids, nonce
    if keypoints[5][2] < conf_threshold or keypoints[6][2] < conf_threshold:
        return "", "", False
    x = (keypoints[5][0] + keypoints[6][0]) / 2
    y = (keypoints[5][1] + keypoints[6][1]) / 2
    threshold = np.linalg.norm(np.array(keypoints[5][:2]) - np.array(keypoints[6][:2]))
    min_dist = float('inf')
    id = None
    label, prob = classify_pose_with_clip(cropped)
    for person in ids:
        if 'prev_x' not in person or 'prev_y' not in person:
            continue
        dist = np.sqrt((x - person['prev_x']) ** 2 + (y - person['prev_y']) ** 2)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            id = person['label']
            label = person['label'] + f" ({label} {prob:.2f})"
            person['prev_x'] = x
            person['prev_y'] = y
            break
    if id is None:
        nonce += 1
        id = "Person_" + str(nonce)
        label = "Person_" + str(nonce) + f" ({label} {prob:.2f})"
        ids.append({'label': "Person_" + str(nonce), 'prev_x': x, 'prev_y': y})
    return id, label, label != "a person not standing"

def cleanup_ids(active_labels):
    global ids
    ids = [person for person in ids if person['label'] in active_labels]

# Phân đoạn người

def segmentation(input_frame, output_frame, seg_model, type = 'Foreground'):
    seg_results = seg_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
    if seg_results[0].masks is not None:
        masks = seg_results[0].masks.data.cpu().numpy()
        combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8) * 255
        combined_mask = cv2.resize(combined_mask, (input_frame.shape[1], input_frame.shape[0]))
        if type == 'Foreground':
            inverse_mask = cv2.bitwise_not(combined_mask)
            output_frame[inverse_mask == 255] = (0, 0, 0)
        elif type == "Binary":
            output_frame = cv2.bitwise_and(output_frame, output_frame, mask=combined_mask)
    return output_frame
