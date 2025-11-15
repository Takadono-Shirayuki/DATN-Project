import cv2
import numpy as np
import threading
import sys
import logging
import os
from recorder import Recorder
from detector import YOLOPersonDetector
from behavior_classifier import BehaviorClassifier
from ultralytics import YOLO
from object_box_lib import separate_object

# Note: write_output and listen_commands removed - this standalone script may need updates
from pose_visualizer import draw_keypoints_on_frame

# Khởi động luồng lệnh từ C#
threading.Thread(target=listen_commands, daemon=True).start()

# Tắt log YOLO và stderr
logging.getLogger('ultralytics').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

# Đọc nguồn video
input = sys.argv[1] if len(sys.argv) > 1 else '0'
if input == '0':
    input = 0
elif input.lower().endswith('.mp4'):
    # Nếu là file mp4 trên máy, sử dụng đường dẫn trực tiếp
    input = input
else:
    input = f'http://{input}:4747/video'
sys.stdout.flush()
cap = cv2.VideoCapture(input)

# initialize recorder with fps from source (fallback 25)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
recorder = Recorder(out_dir='person_videos', fps=fps, frame_size=(224, 224), timeout=5.0)

while not cap.isOpened():
    pass
ret, first_frame = cap.read()
if not ret:
    sys.exit(1)

_, buffer = cv2.imencode('.jpeg', first_frame)
data = buffer.tobytes()
meta_data = { 'loaded_first_frame': True, 'size': len(data), 'object_id': 'first_frame' }
write_output(meta_data, data)

# Tải mô hình
detector = YOLOPersonDetector(model_path='yolov8n.pt', conf_thresh=0.3, use_tracking=True)
seg_model = YOLO('yolov8s-seg.pt')
pose_model = YOLO('yolov8s-pose.pt')

def run_object_box_mode(frame):
    # Phân đoạn ngay trong separate_object để đảm bảo mỗi output chỉ chứa 1 người
    # Truyền seg_model=None để tắt phân đoạn, hoặc seg_model để bật (binary output)
    output_frames = separate_object(
        frame, 
        detector, 
        seg_model=seg_model,
        pose_model=pose_model,
    )

    # Lấy phần tử đầu tiên (nếu có)
    if len(output_frames) == 0:
        output_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        first_id = None
    else:
        first_id = next(iter(output_frames))
        output_frame = output_frames[first_id]['image']

    # Vẽ skeleton lên output_frame
    if first_id is not None and first_id in output_frames:
        keypoints = output_frames[first_id].get('keypoints')
        if keypoints is not None and len(keypoints) > 0:
            output_frame = draw_keypoints_on_frame(output_frame, keypoints)
    
    # Vẽ đường debug (dọc và ngang chính giữa)
    h, w = output_frame.shape[:2]
    cv2.line(output_frame, (w//2, 0), (w//2, h), (0, 0, 255), 1)  # Dọc (đỏ)
    cv2.line(output_frame, (0, h//2), (w, h//2), (0, 0, 255), 1)  # Ngang (đỏ)

    # write per-person video frames
    try:
        for pid, data in output_frames.items():
            img = data.get('image')
            keypoints = data.get('keypoints')
            if img is not None:
                recorder.update(pid, img, keypoints=keypoints)
    except Exception:
        # keep running even if recorder fails for a frame
        pass
    
    _, buffer = cv2.imencode('.jpeg', output_frame)
    data = buffer.tobytes()
    metadata = { 'size': len(data), 'object_id': first_id }
    write_output(metadata, data)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    run_object_box_mode(frame)
cap.release()

# close recorders
try:
    recorder.close_all()
except Exception:
    pass

end_meta_data = { 'end': True }
write_output(end_meta_data, b'')