import cv2
from ultralytics import YOLO
import sys
input = sys.argv[1] if len(sys.argv) > 1 else '0'  # Mặc định sử dụng webcam 0 nếu không có tham số
if input != '0':
    input = 'http://' + input + ':8080/video'
# Tải mô hình YOLOv8 (có thể thay bằng yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
model = YOLO('yolov8s.pt')  # Đảm bảo file .pt đã được tải về hoặc sẽ tự động tải

# Mở luồng video từ IP camera
cap = cv2.VideoCapture(input)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận dạng đối tượng
    results = model.predict(source=frame, save=False, conf=0.5)

    # Vẽ kết quả lên khung hình
    annotated_frame = results[0].plot()
    
    # encode khung hình đã chú thích thành JPEG
    _, buffer = cv2.imencode('.jpeg', annotated_frame)
    data = buffer.tobytes()
    try:
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    except BrokenPipeError:
        print("⚠️ Mất kết nối với tiến trình chính, dừng gửi ảnh.")
        break