import cv2
import sys
import logging
import os
import threading
from ultralytics import YOLO
from yolo_detect_lib import *

def run_camera_mode(frame):
    output_frame = frame.copy()

    # Phân đoạn người
    if State.Camera.enable_segmentation:
        output_frame = segmentation(frame, output_frame, seg_model, type='Foreground')

    # Nhận diện bbox và keypoints
    if State.Camera.enable_pose or State.Camera.enable_bbox:
        output_frame = detect_pose(frame, output_frame, pose_model)

    # Gửi ảnh về C#
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data), 'mode': State.mode }
    write_output(metadata, data)

def run_object_box_mode(frame):
    output_frames = separate_object(frame, pose_model)

    if State.ObjectBox.tracking_id not in output_frames:
        output_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        output_frame = output_frames[State.ObjectBox.tracking_id]['image']
        cv2.putText(output_frame, output_frames[State.ObjectBox.tracking_id]['label'], (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if State.ObjectBox.enable_segmentation:
        for id in output_frames:
            output_frames[id]['image'] = segmentation(output_frames[id]['image'], output_frames[id]['image'], seg_model, type='Foreground')

    # Gửi ảnh về C#
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data), 'mode': State.mode }
    write_output(metadata, data)

# Khởi động luồng lệnh từ C#
threading.Thread(target=listen_commands, daemon=True).start()

# Tắt log YOLO và stderr
logging.getLogger('ultralytics').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

# Đọc nguồn video
input = sys.argv[1] if len(sys.argv) > 1 else '0'
if input == '0':
    input = 0
else:
    input = 'http://192.168.1.136:4747/video'
cap = cv2.VideoCapture(input)

# Đọc chế độ
State.mode = sys.argv[2] if len(sys.argv) > 2 else "camera"

# Tải mô hình
pose_model = YOLO('yolov8s-pose.pt')
seg_model = YOLO('yolov8s-seg.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if State.mode == "camera":
        run_camera_mode(frame)
    elif State.mode == "object_box":
        run_object_box_mode(frame)
cap.release()
