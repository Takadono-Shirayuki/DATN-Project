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
    enable_bbox = True
    enable_pose = False
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
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Danh sách mô tả tư thế
text_labels = [
    "a person walking",
    "a person running",
    "a person standing",
    "a person not standing",
]
text_tokens = clip.tokenize(text_labels).to(device)

# Hàm phân loại tư thế
def classify_pose_with_clip(image_np):
    # Chuyển ảnh từ numpy sang PIL
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_preprocessed = preprocess(image_pil).unsqueeze(0).to(device)

    # Tính độ tương đồng
    with torch.no_grad():
        image_features = clip_model.encode_image(image_preprocessed)
        text_features = clip_model.encode_text(text_tokens)

        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits[0].cpu().numpy()

    # Trả về nhãn có xác suất cao nhất
    best_index = probs.argmax()
    return text_labels[best_index], probs[best_index]

# Lệnh từ C#
def listen_commands():
    global State
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "bbox_on": State.enable_bbox = True
        elif cmd == "bbox_off": State.enable_bbox = False
        elif cmd == "pose_on": State.enable_pose = True
        elif cmd == "pose_off": State.enable_pose = False
        elif cmd == "seg_on": State.enable_segmentation = True
        elif cmd == "seg_off": State.enable_segmentation = False

# Gán ID
ids = []
nonce = 0

def assign_id_for_person(keypoints, cropped, conf_threshold=0.3):
    global ids, nonce

    # Kiểm tra keypoint vai
    if keypoints[5][2] < conf_threshold or keypoints[6][2] < conf_threshold:
        return "", "", False

    # Tính trung điểm vai
    x = (keypoints[5][0] + keypoints[6][0]) / 2
    y = (keypoints[5][1] + keypoints[6][1]) / 2
    threshold = np.linalg.norm(np.array(keypoints[5][:2]) - np.array(keypoints[6][:2]))

    # Tìm người gần nhất trong danh sách đã gán
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

    # Nếu không khớp ai → gán ID mới
    if id is None:
        nonce += 1
        id = "Person_" + str(nonce)
        label = "Person_" + str(nonce) + f" ({label} {prob:.2f})"
        ids.append({'label': "Person_" + str(nonce), 'prev_x': x, 'prev_y': y})

    return id, label, label != "a person not standing"

def cleanup_ids(active_labels):
    global ids
    ids = [person for person in ids if person['label'] in active_labels]

# Vẽ keypoints và khung xương
def draw_pose(frame, keypoints):
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
                    cv2.putText(frame, f'{c:.2f}', (int(x) + 6, int(y) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if 5 in drawed_keypoints and 6 in drawed_keypoints:
        x_mid = int((keypoints[5][0] + keypoints[6][0]) // 2)
        y_mid = int((keypoints[5][1] + keypoints[6][1]) // 2)
        conf_mid = (keypoints[5][2] + keypoints[6][2]) / 2
        cv2.circle(frame, (x_mid, y_mid), 6, (255, 0, 0), -1)
        cv2.putText(frame, f'{conf_mid:.2f}', (x_mid + 6, y_mid - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.line(frame, (int(keypoints[0][0]), int(keypoints[0][1])), (x_mid, y_mid), (0, 255, 255), 2)

# Gửi dữ liệu về C#
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

# Nhận diện hộp giới hạn và khung xương
def detect_pose(input_frame, output_frame, pose_model):
    active_labels = set()
    pose_results = pose_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
    for box, kp in zip(pose_results[0].boxes.xyxy, pose_results[0].keypoints.data):
        keypoints = [[round(x, 2), round(y, 2), round(c, 2)] for x, y, c in kp.tolist()]
        cropped = input_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        id, label, moving = assign_id_for_person(keypoints, cropped)
        if not moving:
            continue
        active_labels.add(id)
        if State.enable_bbox:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        if State.enable_pose:
            draw_pose(output_frame, keypoints)
    if len(active_labels) > 0:
        count = sum(1 for file in os.listdir('AI_model/temp') if file.startswith('Person_') and file.endswith('.jpg'))
        cv2.imwrite(f'AI_model/temp/Person_{count}.jpg', output_frame)
    cleanup_ids(active_labels)
    return output_frame