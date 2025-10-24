import cv2
import numpy as np
import threading
import sys
import os
import logging
from ultralytics import YOLO
from common import State, write_output, segmentation, listen_commands
from object_box_lib import separate_object

# Khởi động luồng lệnh từ C#
threading.Thread(target=listen_commands, daemon=True).start()

# Tắt log YOLO và stderr
logging.getLogger('ultralytics').setLevel(logging.ERROR)
# sys.stderr = open(os.devnull, 'w')

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
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data) }
    write_output(metadata, data)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    run_object_box_mode(frame)
cap.release()
