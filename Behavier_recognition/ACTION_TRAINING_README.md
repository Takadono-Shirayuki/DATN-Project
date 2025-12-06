# Action Recognition Training

Hệ thống training và inference cho mô hình nhận diện hành động (walking vs non-walking) từ keypoints skeleton.

## 📁 Cấu trúc dữ liệu

```
dataset/
  ├── walking/           # Video người đi bộ
  │   ├── video1.json
  │   ├── video2.json
  │   └── ...
  └── non-walking/       # Video người không đi bộ (đứng, ngồi, v.v.)
      ├── video3.json
      ├── video4.json
      └── ...
```

**Format JSON** (đầu ra từ recorder):
```json
{
  "fps": 25.0,
  "frame_size": [224, 224],
  "total_frames": 100,
  "duration": 4.0,
  "frames": [
    [[x1, y1, conf1], [x2, y2, conf2], ...],  // Frame 1 (17 keypoints)
    [[x1, y1, conf1], [x2, y2, conf2], ...],  // Frame 2
    ...
  ]
}
```

## 🚀 Training

### 1. Classification Model (LSTM-based)
Mô hình phân loại cơ bản sử dụng LSTM:

```bash
python train_action_classifier.py
```

**Hyperparameters:**
- `SEQUENCE_LENGTH = 30`: Số frame mỗi sequence (1.2 giây @ 25fps)
- `STRIDE = 15`: Bước trượt sliding window (overlap 50%)
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 50`
- `LEARNING_RATE = 0.001`
- `HIDDEN_SIZE = 128`
- `NUM_LAYERS = 2`

**Output:**
- `action_classifier_best.pth`: Model tốt nhất
- `training_history_classifier.png`: Biểu đồ training

### 2. Contrastive Learning Model (SimCLR-style)
Mô hình sử dụng contrastive learning để học representation tốt hơn:

```bash
python train_action_contrastive.py
```

**Training Process:**
1. **Phase 1 (60% epochs)**: Contrastive pre-training
   - Học representation từ augmented pairs
   - Không cần label
   
2. **Phase 2 (40% epochs)**: Fine-tuning with classification
   - Combined loss: 30% contrastive + 70% classification
   - Fine-tune cả encoder và classifier

**Hyperparameters:**
- `NUM_EPOCHS = 100` (60 pre-train + 40 fine-tune)
- `TEMPERATURE = 0.5`: Temperature for contrastive loss
- Các tham số khác giống classifier model

**Data Augmentation:**
- Random temporal shift
- Random noise
- Random horizontal flip
- Random scaling

**Output:**
- `action_contrastive_best.pth`: Model tốt nhất
- `training_history_contrastive.png`: Biểu đồ training

## 🔮 Inference

### Single file prediction:
```bash
python inference_action.py action_classifier_best.pth dataset/walking/video1.json
```

### Batch prediction (folder):
```bash
python inference_action.py action_contrastive_best.pth dataset/walking/
```

**Output example:**
```
File                                     Prediction      Confidence
======================================================================
video1.json                              Walking         95.32%
video2.json                              Non-Walking     87.64%
```

## 📊 Model Comparison

| Model | Architecture | Training Time | Accuracy | Best Use Case |
|-------|-------------|---------------|----------|---------------|
| **Classifier** | Bi-LSTM + FC | Faster (~30 min) | Good | Simple tasks, small data |
| **Contrastive** | SimCLR + Bi-LSTM | Slower (~60 min) | Better | Complex actions, more data |

## 🎯 Key Parameters

### Sequence Length
- **30 frames** (1.2s @ 25fps): Đủ để phát hiện 1 bước chân
- Tăng lên nếu cần context dài hơn (vd: 60 frames = 2.4s)

### Stride
- **15 frames** (overlap 50%): Cân bằng giữa data augmentation và tốc độ
- Stride nhỏ hơn = nhiều data hơn nhưng training chậm hơn
- Stride = sequence_length = không overlap

### Hidden Size
- **128**: Đủ cho skeleton 17 keypoints
- Tăng nếu cần model phức tạp hơn (256, 512)

## 💡 Tips

1. **Data Balance**: Cần số lượng tương đương giữa walking và non-walking
2. **Quality Check**: Xóa video < 2s hoặc keypoints không đủ
3. **Augmentation**: Contrastive model tận dụng augmentation tốt hơn
4. **GPU**: Khuyến khích dùng GPU (nhanh hơn 5-10x)
5. **Early Stopping**: Model tự động save best validation accuracy

## 📝 Notes

- Input: 17 keypoints COCO format (x, y, confidence)
- Normalization: Tọa độ chia cho 224 (frame size)
- Label: 0 = non-walking, 1 = walking
- Framework: PyTorch

## 🐛 Troubleshooting

**Lỗi: No data found**
```
Kiểm tra cấu trúc thư mục dataset/walking và dataset/non-walking
Đảm bảo có file .json trong các thư mục này
```

**Lỗi: CUDA out of memory**
```
Giảm BATCH_SIZE xuống 16 hoặc 8
Giảm HIDDEN_SIZE xuống 64
```

**Accuracy thấp**
```
Tăng NUM_EPOCHS (classifier: 100, contrastive: 150)
Tăng SEQUENCE_LENGTH (60 frames)
Kiểm tra chất lượng dữ liệu (keypoints có đầy đủ không?)
Thử contrastive model thay vì classifier
```
