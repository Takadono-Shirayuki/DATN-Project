import cv2
import sys
import logging
import json
from ultralytics import YOLO
import os

# Tắt log của YOLO
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Chặn stderr tạm thời
sys.stderr = open(os.devnull, 'w')

# Đọc nguồn video
input = sys.argv[1] if len(sys.argv) > 1 else 0
if input != 0:
    input = 'http://' + str(input) + ':8080/video'

# Mở luồng video
cap = cv2.VideoCapture(input)

pose_model = YOLO('yolov8s-pose.pt')  # Nhận diện keypoints trực tiếp

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pose_results = pose_model.predict(source=frame, save=False, conf=0.5, verbose=False)

    person_boxes = []
    all_keypoints = []

    # Truy cập đúng cấu trúc: [num_people, num_keypoints, 3]
    for box, kp in zip(pose_results[0].boxes.xyxy, pose_results[0].keypoints.data):
        # Bounding box
        xyxy = box.tolist()
        person_boxes.append([round(coord, 2) for coord in xyxy])

        # Keypoints
        keypoints = []
        for point in kp:
            x, y, conf = point.tolist()
            keypoints.append([round(x, 2), round(y, 2), round(conf, 2)])
        all_keypoints.append(keypoints)

    # Vẽ khung (bounding box) quanh người
    for box in pose_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mã hóa JPEG
    _, buffer = cv2.imencode('.jpeg', frame)
    data = buffer.tobytes()

    # Tạo metadata
    metadata = {
        'size': len(data),
        'detections': person_boxes,
        'keypoints': all_keypoints
    }
    meta_json = json.dumps(metadata).encode('utf-8')
    
    try:
        # Gửi metadata trước
        sys.stdout.buffer.write(b'--META--\n')
        sys.stdout.buffer.write(meta_json)
        sys.stdout.buffer.write(b'\n--ENDMETA--\n')

        # Gửi ảnh JPEG
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.write(b'\n')
        sys.stdout.flush()
    except BrokenPipeError:
        print("⚠️ Mất kết nối với tiến trình chính, dừng gửi ảnh.")
        break

cap.release()
