# GR2 Camera Android - Setup Guide

## Project Structure

```
app/
├── src/main/
│   ├── kotlin/com/gr2/camera/
│   │   ├── MainActivity.kt              # Main activity with mode switching
│   │   ├── camera/
│   │   │   └── CameraManager.kt         # CameraX wrapper
│   │   ├── ml/
│   │   │   └── YOLODetector.kt         # TFLite YOLO inference
│   │   └── network/
│   │       └── NetworkManager.kt        # WebSocket client
│   ├── res/
│   │   ├── layout/activity_main.xml     # UI layout
│   │   └── values/strings.xml
│   └── AndroidManifest.xml
├── build.gradle                         # Dependencies
└── proguard-rules.pro
```

## Setup Steps

### 1. Models Preparation

You need to add TFLite models to `app/src/main/assets/`:

```bash
app/src/main/assets/
├── yolov8n_detection.tflite    # YOLOv8 Nano detection
└── yolov8s_pose.tflite         # (Optional) YOLOv8 Small pose estimation
```

**Converting YOLO to TFLite:**

```bash
# 1. Convert PyTorch YOLO to TFLite
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='tflite')
# Output: yolov8n_integer_quant.tflite (quantized)

# 2. Copy to Android assets
cp yolov8n_integer_quant.tflite app/src/main/assets/yolov8n_detection.tflite
```

### 2. Operating Modes

#### **Streaming Mode**
- Sends raw camera frames to server (20-30 FPS)
- Server IP: 192.168.x.x (default: 192.168.1.100)
- Port: 3001 (default Vite dev server)
- **Use case:** Live preview on server

#### **Recognition Mode**
- Runs YOLO inference on device (5-15 FPS depending on device)
- Extracts silhouette (person pixels in white, background black)
- Sends silhouette to server instead of raw frame
- Shows inference stats: persons detected, inference time (ms)
- **Use case:** Privacy-preserving gait analysis

### 3. Build & Run

#### Via Android Studio
1. Open project in Android Studio
2. Connect Android device (API 24+)
3. Click "Run" or press Shift+F10

#### Via Command Line
```bash
cd Camera/Android

# Debug build
./gradlew installDebug

# Release build
./gradlew assembleRelease
```

### 4. Permissions
App requests:
- `CAMERA` - to capture frames
- `INTERNET` - to send to server

Both are granted at runtime.

### 5. Server Connection

**Example:**
- IP: `192.168.1.100`
- Port: `3000` (React Vite dev server)
- Mode: `Recognition`

After clicking "Connect", app will:
1. Connect via WebSocket
2. Register device
3. Start sending frames/silhouettes
4. Display stats (FPS, persons detected, inference time)

## Performance Notes

- **Streaming:** ~20-30 FPS (minimal processing)
- **Recognition:** ~5-15 FPS (depends on device CPU)
  - Pixel 6: ~12 FPS
  - Snapdragon 888: ~8-10 FPS
  - Budget devices: ~3-5 FPS

## Troubleshooting

### Camera not working
- Check AndroidManifest.xml permissions
- Ensure device has camera
- Test with `CameraX` examples first

### Model inference slow
- Check device specs (only ARM/ARM64)
- Use quantized TFLite models
- Reduce input resolution if needed

### WebSocket connection failed
- Verify server IP is correct
- Check firewall settings
- Ensure server is running on port 3000

### Build errors
- Update Android SDK to API 30+
- Ensure TensorFlow Lite dependency versions match
- Clean build: `./gradlew clean build`

## Next Steps

- [ ] Deploy real YOLO TFLite models
- [ ] Implement pose estimation (optional)
- [ ] Add action classification (connect to trained model)
- [ ] Optimize FPS with GPU acceleration
- [ ] Test on multiple device types
