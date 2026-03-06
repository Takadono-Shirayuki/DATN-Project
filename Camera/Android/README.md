# Gait Recognition - Android Camera App

## Tổng quan

Ứng dụng Android cho phép:
1. **Chế độ Camera**: Lấy stream camera thẳng gửi đến server (cần IP server)
2. **Chế độ Nhận diện**: Xử lý hình ảnh tại thiết bị (YOLO + Action Classifier) trước khi gửi

## Cấu trúc dự án

```
Android/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── kotlin/
│   │   │   │   └── com/gr2/camera/
│   │   │   │       ├── MainActivity.kt
│   │   │   │       ├── ui/
│   │   │   │       ├── camera/
│   │   │   │       ├── network/
│   │   │   │       └── models/
│   │   │   ├── res/
│   │   │   └── AndroidManifest.xml
│   │   └── test/
│   ├── build.gradle
│   └── proguard-rules.pro
├── build.gradle
├── settings.gradle
└── README.md
```

## Tính năng chính

### 1. Chế độ Camera (Live Streaming)
- Nhập IP server
- Stream MJPEG hoặc H264
- Kết nối qua Socket/WebSocket

### 2. Chế độ Nhận diện
- YOLOv8 nano (phát hiện người)
- YOLOv8 pose (lấy skeleton)
- Action Classifier (phân loại hành động)
- GEI Generator (tạo ảnh bóng)
- Gửi GEI + metadata đến server

## Công nghệ

- **Kotlin** - Ngôn ngữ chính
- **OpenCV** - Xử lý hình ảnh
- **TensorFlow Lite** - Chạy mô hình
- **OkHttp** - HTTP client
- **Jetpack** - Android components

## TODO

- [ ] Thiết lập project Gradle cơ bản
- [ ] Màn hình quay video
- [ ] Đăng ký server
- [ ] Chế độ camera streaming
- [ ] Chế độ nhận diện
- [ ] Gửi dữ liệu đến server
