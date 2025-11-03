import cv2
import numpy as np
import threading
import sys
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
elif input.lower().endswith('.mp4'):
    # Nếu là file mp4 trên máy, sử dụng đường dẫn trực tiếp
    input = input
else:
    input = f'http://{input}:4747/video'
cap = cv2.VideoCapture(input)


# Tải mô hình
pose_model = YOLO('yolov8s-pose.pt')
seg_model = YOLO('yolov8s-seg.pt')

def run_object_box_mode(frame):
    output_frames = separate_object(frame, pose_model)
    # Nếu tracking_id không tồn tại, trả về phần tử đầu tiên (nếu có)
    if State.ObjectBox.tracking_id not in output_frames:
        if len(output_frames) == 0:
            output_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Lấy phần tử đầu tiên trong dict (an toàn với Python 3.7+ giữ thứ tự chèn)
            first_id = next(iter(output_frames))
            output_frame = output_frames[first_id]['image']
    else:
        output_frame = output_frames[State.ObjectBox.tracking_id]['image']
    if State.ObjectBox.enable_segmentation:
        for id in output_frames:
            output_frames[id]['image'] = segmentation(output_frames[id]['image'], output_frames[id]['image'], seg_model, type='Foreground')
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data), 'object_id': State.ObjectBox.tracking_id if State.ObjectBox.tracking_id in output_frames else first_id }
    write_output(metadata, data)
    cv2.imshow('Object Box', output_frame)
    cv2.waitKey(1)
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    run_object_box_mode(frame)
cap.release()

end_meta_data = { 'end': True }
write_output(end_meta_data, b'')