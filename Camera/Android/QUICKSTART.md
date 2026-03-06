# 🚀 Android App - Quick Start

## 5-Minute Setup

### Step 1: Convert YOLO Model
```bash
# From project root
cd Camera/Android
python convert_to_tflite.py --input ../../archive/yolov8n.pt --output app/src/main/assets/

# Verify
ls -la app/src/main/assets/yolov8n*.tflite
```

### Step 2: Open in Android Studio
```bash
# Method 1: Command line
./gradlew assembleDebug

# Method 2: Android Studio
1. File → Open → GR2-Project/Camera/Android
2. Wait for Gradle sync
3. Plugin Android device
4. Run → app
```

### Step 3: Test Connection
1. **Server IP:** Your machine IP (e.g., 192.168.1.100)
2. **Port:** 3001 (React Vite default)
3. **Mode:** Recognition
4. **Click:** Connect

### Step 4: Monitor
Watch stats update in real-time:
```
Status: Connected - recognition
Frames: 45 | FPS: 12 | Persons: 1 | Inference: 85ms
```

## 📱 Features at a Glance

| Feature | Streaming | Recognition |
|---------|-----------|-------------|
| Live camera preview | ✅ | ✅ |
| Send to server | ✅ Raw frames | ✅ Silhouettes |
| AI inference | ❌ | ✅ YOLO TFLite |
| FPS target | 20-30 | 10-15 |
| Latency | ~50ms | ~100-150ms |
| Battery drain | Low | Medium |

## 🔌 Server Integration

App sends WebSocket messages:

**Streaming Frame:**
```json
{
  "type": "frame",
  "frame": "base64encodedJPEG...",
  "metadata": {
    "mode": "streaming",
    "timestamp": 1709754000000
  }
}
```

**Recognition Frame (Silhouette):**
```json
{
  "type": "frame",
  "frame": "base64silhouettePNG...",
  "metadata": {
    "mode": "recognition",
    "detections": 1,
    "inference_ms": 85,
    "pipeline": "yolo_tflite_silhouette",
    "action": "unknown",
    "timestamp": 1709754000000
  }
}
```

## ⚙️ Configuration

### app/build.gradle
- Minimum API: 24 (Android 7.0)
- Target API: 34 (Android 14)
- Kotlin 1.9.0
- Gradle 8.1.0

### Permissions (auto-granted at runtime)
- CAMERA - capture video
- INTERNET - WebSocket

### Storage
- Models: `app/src/main/assets/` (~7-8MB)
- Cache: `/cache/` (temporary JPEG encoding)

## 🐛 Common Issues

**Q: "Model not found" error**
A: Copy `.tflite` file to `app/src/main/assets/`

**Q: Apps drops connectivity**
A: Add permission in AndroidManifest.xml:
```xml
<uses-permission android:name="android.permission.CHANGE_NETWORK_STATE" />
```

**Q: FPS too low in recognition**
A: 
- Try landscape orientation
- Close background apps
- Use fewer cameras permissions

**Q: Can't connect to server**
A:
- Verify IP reachable: `ping 192.168.1.100`
- Check server listening: `netstat -an | grep 3001`
- Firewall: Allow port 3001

## 📚 Full Documentation

See:
- [SETUP.md](./SETUP.md) - Detailed setup
- [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) - Architecture details
- [README.md](./README.md) - Project overview

## 🎯 What's Next?

After getting streaming/recognition working:

1. **Pose Estimation** - Add keypoint detection (optional)
2. **Action Classification** - Connect trained action classifier model
3. **GEI Generation** - Create gait energy images from silhouettes
4. **Local Storage** - Save frames locally for offline analysis
5. **GPU Acceleration** - Enable TFLite GPU delegate for 2x speed

---

**Happy coding! 🎉**
