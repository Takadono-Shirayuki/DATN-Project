import cv2
from common import State, write_output, segmentation, listen_commands
from camera_lib import detect_pose
from ultralytics import YOLO
import threading
import sys
import os
import logging

# Khởi động lệnh từ C#
threading.Thread(target=listen_commands, daemon=True).start()

logging.getLogger('ultralytics').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

# Đọc nguồn video
input = sys.argv[1] if len(sys.argv) > 1 else '0'
if input == '0':
    input = 0
else:
    input = f'http://{input}:4747/video'
cap = cv2.VideoCapture(input)


# Tải mô hình
pose_model = YOLO('yolov8s-pose.pt')
seg_model = YOLO('yolov8s-seg.pt')

def run_camera_mode(frame):
    output_frame = frame.copy()
    if State.Camera.enable_segmentation:
        output_frame = segmentation(frame, output_frame, seg_model, type='Foreground')
    if State.Camera.enable_pose or State.Camera.enable_bbox:
        output_frame = detect_pose(frame, output_frame, pose_model)
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data), 'segmentation': State.Camera.enable_segmentation, 'pose': State.Camera.enable_pose, 'bbox': State.Camera.enable_bbox}
    write_output(metadata, data)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    run_camera_mode(frame)
cap.release()
