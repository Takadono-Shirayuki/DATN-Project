import cv2
import sys
import logging
import os
import threading
from ultralytics import YOLO
from yolo_detect_lib import *

# Khởi động luồng lệnh từ C#
threading.Thread(target=listen_commands, daemon=True).start()

# Tắt log YOLO và stderr
logging.getLogger('ultralytics').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

# Đọc nguồn video
input = sys.argv[1] if len(sys.argv) > 1 else 0
if input != 0:
    input = 'http://' + str(input) + ':8080/video'
cap = cv2.VideoCapture(input)

# Tải mô hình
pose_model = YOLO('yolov8s-pose.pt')
seg_model = YOLO('yolov8s-seg.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()

    # Phân đoạn người
    if State.enable_segmentation:
        output_frame = segmentation(frame, output_frame, seg_model, type='Foreground')

    # Nhận diện bbox và keypoints
    if State.enable_pose or State.enable_bbox:
        output_frame = detect_pose(frame, output_frame, pose_model)

    # Gửi ảnh về C#
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data) }
    write_output(metadata, data)
cap.release()
