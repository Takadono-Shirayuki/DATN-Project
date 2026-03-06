# Android App - Recognition Pipeline Implementation

## ✅ Completed Components

### 1. **MainActivity.kt** - Main Activity
- ✅ Dual-mode operation: Streaming vs Recognition
- ✅ Real-time frame capture using ImageProxy  
- ✅ Status bar showing connection state
- ✅ Statistics display (FPS, persons detected, inference time)
- ✅ Connect/Disconnect button with mode toggle

### 2. **CameraManager.kt** - Camera Control
- ✅ CameraX integration for modern camera API
- ✅ Real-time frame analysis with ImageAnalysis
- ✅ Background thread for image processing
- ✅ Callback-based frame delivery
- ✅ Back camera support

### 3. **YOLODetector.kt** - AI Inference
- ✅ TensorFlow Lite YOLO model loading
- ✅ Person detection with confidence threshold
- ✅ Input preprocessing (640x640 resize, normalization)
- ✅ Output parsing and NMS
- ✅ SilhouetteExtractor class for background detection

### 4. **NetworkManager.kt** - Server Communication
- ✅ WebSocket client with OkHttp3
- ✅ Automatic device registration
- ✅ Frame encoding to JPEG Base64
- ✅ Metadata attachment (persons, inference time, mode)
- ✅ Connection state tracking
- ✅ Error handling and reconnection

### 5. **UI Layout** (activity_main.xml)
- ✅ Camera preview (full-screen capable)
- ✅ Status TextView (color-coded: green=connected, red=error)
- ✅ Statistics TextView (FPS, persons, inference ms)
- ✅ Server IP input field
- ✅ Server Port input field  
- ✅ Radio buttons: Streaming vs Recognition mode
- ✅ Connect/Disconnect button

## 🔄 Operating Modes

### **Streaming Mode**
```
Camera Frame → PreviewView → Server (WebSocket)
FPS: 20-30 (minimal processing)
Sends: Raw JPEG frames
Use: Live preview monitoring
```

### **Recognition Mode** 
```
Camera Frame → YOLO Inference → Silhouette Extraction → Server (WebSocket)
FPS: 5-15 (device dependent)
Sends: Silhouette bitmap (person=white, background=black)
Use: Privacy-preserving gait analysis
```

## 📦 Dependencies Added

```gradle
// TensorFlow Lite (2.13.0)
- tensorflow-lite
- tensorflow-lite-support
- tensorflow-lite-gpu (optional)

// Camera (AndroidX 1.3.0)
- androidx.camera:camera-core
- androidx.camera:camera-view
- androidx.camera:camera-lifecycle

// Network
- okhttp3:okhttp (4.11.0)
- org.json:json (JSON parsing)

// UI & Lifecycle
- androidx.appcompat:appcompat (1.6.1)
- androidx.lifecycle:lifecycle-runtime-ktx (2.6.1)
```

## 🚀 Next Steps - Model Setup

### 1. Convert YOLO to TFLite
```bash
cd Camera/Android
python convert_to_tflite.py --input ../../archive/yolov8n.pt --output app/src/main/assets/
```

### 2. Verify Assets
```bash
ls -la app/src/main/assets/
# Should contain: yolov8n_detection.tflite (~6-8 MB)
```

### 3. Build & Deploy
```bash
./gradlew assembleDebug
# or use Android Studio's "Run" button
```

### 4. Connect to Server
- Enter server IP (e.g., 192.168.1.100)
- Port: 3001 (React Vite dev)
- Select mode: Recognition
- Click "Connect"
- Monitor stats in real-time

## 📊 Expected Performance

| Device | Streaming FPS | Recognition FPS | Inference Time |
|--------|--------------|-----------------|-----------------|
| Pixel 6 | 25-30 | 10-12 | 80-100ms |
| Snapdragon 888 | 28-30 | 8-10 | 100-120ms |
| Budget (SD 6xx) | 20-25 | 3-5 | 200-300ms |

## 🔧 Troubleshooting

### Camera not displaying
- Check AndroidManifest.xml has CAMERA permission
- Verify device has working camera
- In emulator, enable virtual camera

### Model not loading
- Verify `yolov8n_detection.tflite` exists in `assets/`
- Check file size (should be 5-10 MB)
- Ensure TFLite dependency version matches

### WebSocket not connecting
- Verify server IP is reachable from device
- Check port 3001 is open/forwarded
- Server must accept WebSocket upgrade at `/camera`

### Low FPS in recognition mode
- Close other apps consuming CPU
- Try reducing input resolution in YOLODetector
- Use quantized model for faster inference

## 📝 Code Summary

- **Total Kotlin classes:** 4 (MainActivity, CameraManager, YOLODetector, NetworkManager)
- **Total files:** 8 main source files
- **Lines of code:** ~800 lines of Kotlin
- **Architecture:** MVVM-lite with callback pattern
- **Thread model:** Main UI + Background analysis threads
