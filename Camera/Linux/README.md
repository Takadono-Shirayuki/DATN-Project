# Gait Recognition - Linux/Desktop Camera App

## Tổng quan

Ứng dụng Python cho Linux/Desktop cho phép:
1. **Chế độ Camera**: Lấy stream camera thẳng gửi đến server (cần IP server)
2. **Chế độ Nhận diện**: Xử lý hình ảnh tại thiết bị (YOLO + Action Classifier + GEI) trước khi gửi

## Cấu trúc dự án

```
Linux/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── camera_capture.py
│   │   └── stream_handler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py
│   │   ├── pose_estimator.py
│   │   ├── action_classifier.py
│   │   └── gei_generator.py
│   ├── network/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── protocol.py
│   └── ui/
│       ├── __init__.py
│       └── display.py
├── models/
│   ├── yolov8n.pt (hoặc từ archive)
│   ├── yolov8s-pose.pt
│   └── action_classifier.pth
├── config.yaml
├── requirements.txt
└── README.md
```

## Tính năng chính

### 1. Chế độ Camera (Live Streaming)
```python
python main.py --mode streaming --server-ip 192.168.x.x --server-port 3000
```
- Nhập IP/Port server
- Stream MJPEG hoặc raw frames
- Kết nối qua Socket TCP/WebSocket

### 2. Chế độ Nhận diện
```python
python main.py --mode recognition --server-ip 192.168.x.x
```
- YOLOv8 nano (phát hiện người)
- YOLOv8 pose (lấy skeleton 17 điểm)
- Action Classifier (phân loại hành động)
- GEI Generator (tạo Gait Energy Image)
- Gửi GEI + metadata (action, confidence) đến server

## Công nghệ

- **Python 3.8+** - Ngôn ngữ chính
- **OpenCV** - Xử lý hình ảnh
- **YOLOv8** - Phát hiện & pose
- **PyTorch** - Action classifier & GEI encoder
- **Requests/WebSocket** - Giao tiếp mạng
- **Tkinter/PyQt** - GUI tùy chọn

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

### Chế độ Camera
```bash
python src/main.py --mode streaming --server-ip 192.168.1.100
```

### Chế độ Nhận diện
```bash
python src/main.py --mode recognition --server-ip 192.168.1.100
```

## TODO

- [ ] Thiết lập camera capture
- [ ] Tích hợp YOLO detection
- [ ] Tích hợp pose estimation
- [ ] Tích hợp action classifier
- [ ] Tích hợp GEI generator
- [ ] Tích hợp network client
- [ ] Chế độ streaming
- [ ] Chế độ recognition
- [ ] GUI đơn giản
