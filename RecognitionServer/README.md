# Recognition Server

Server Python xử lý nhận diện và tạo silhouette từ hình ảnh camera.

## Chức năng

- Nhận hình ảnh thô (raw frames) từ app Android qua WebSocket (port **3001**)
- Chạy YOLO inference để phát hiện người
- Tạo silhouette (ảnh bóng) từ các bounding box
- Gửi kết quả đến UI Server (port 3000)

## Cài đặt

```bash
# Cài dependencies
pip install -r requirements.txt

# Tải YOLO model (chỉ cần 1 lần)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Chạy Server

### Cách 1: Mặc định (port 3001, UI server localhost:3000)

```bash
python server.py
```

### Cách 2: Custom port và UI server URL

```bash
python server.py 3001 http://192.168.1.100:3000
```

**Tham số**:
- Tham số 1: Port lắng nghe WebSocket (mặc định 3001)
- Tham số 2: URL của UI server để forward kết quả (mặc định http://localhost:3000)

## Cấu hình Android App

Khi chọn mode **Recognition** trong app Android:

1. **Server IP**: Nhập IP của máy chạy server này (VD: 192.168.1.100)
2. **Server Port**: Nhập **3000** (base port)
3. App sẽ tự động kết nối đến port **3001** (base port + 1)

## Luồng xử lý

```
Android App (Recognition Mode)
    │
    ├─ Capture camera frame
    ├─ Encode to JPEG + base64
    └─ Send via WebSocket
        │
        ▼
Recognition Server (Port 3001)
    │
    ├─ Decode base64 → numpy array
    ├─ YOLO Inference (~120ms)
    │   └─ Detect persons (class 0)
    ├─ Extract Silhouette (~5ms)
    │   └─ White boxes on black canvas
    ├─ Encode silhouette to base64
    └─ POST to UI Server
        │
        ▼
UI Server (Port 3000)
    │
    ├─ Receive via /api/frames
    ├─ Store in memory
    └─ Broadcast to React clients
        │
        ▼
React UI
    └─ Display silhouette image
```

## Hiệu suất

| Thiết bị | Inference Time | FPS |
|----------|---------------|-----|
| CPU (i5/i7) | 100-150ms | 7-10 |
| GPU (RTX 3060) | 20-40ms | 25-50 |

**Tăng tốc với GPU**:

Sửa trong `server.py` dòng:
```python
self.model.to('cpu')  # Đổi thành 'cuda'
```

## Message Format

### Input (từ Android)

```json
{
  "type": "frame",
  "frame": "iVBORw0KGgoAAAANSUhEUg...",
  "metadata": {}
}
```

### Output (gửi đến UI Server)

```json
{
  "type": "frame",
  "frame": "base64_silhouette_image",
  "mode": "recognition",
  "metadata": {
    "person_count": 1,
    "inference_ms": 125.5,
    "timestamp": "2026-03-06T10:30:45.123456",
    "device": "192.168.1.50"
  }
}
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'websockets'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: yolov8n.pt"
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').info()"
```

### Connection Refused
- Kiểm tra firewall cho phép port 3001
- Kiểm tra IP address đúng
- Test: `telnet <server_ip> 3001`

### Inference chậm
- Dùng GPU: đổi `.to('cpu')` → `.to('cuda')`
- Giảm độ phân giải camera trên Android
- Giảm confidence threshold (0.4 → 0.3)

## Logs

Server in log ra console:

```
2026-03-06 10:30:45 - INFO - Loading YOLOv8n model...
2026-03-06 10:30:46 - INFO - Model loaded successfully
2026-03-06 10:30:46 - INFO - Recognition server listening on ws://0.0.0.0:3001
2026-03-06 10:31:00 - INFO - Client connected: ('192.168.1.50', 54321)
2026-03-06 10:31:30 - INFO - Processed 30 frames | FPS: 8.5 | Persons: 1 | Inference: 118.3ms
```

## Tích hợp với hệ thống

Server này là một phần của hệ thống GR2:

```
RecognitionServer (Port 3001) ← Android app (recognition mode)
                ↓
         UI Server (Port 3000) ← React Frontend
```

Xem thêm:
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Tổng quan kiến trúc
- [UI/server.js](../UI/server.js) - UI server nhận kết quả
- [Camera/Android](../Camera/Android/) - Android app source code
