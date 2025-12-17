import torch
import torch.nn as nn
from torchvision import models

# Cấu hình chung
SEQ_LEN = 32
EMBEDDING_DIM = 128
IMG_SIZE = 224

# 1. ENCODER CỐT LÕI (ResNet + LSTM)
# Phần này sẽ được lưu ở Giai đoạn 1 và tái sử dụng ở Giai đoạn 2
class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        # Backbone: ResNet18 (bỏ lớp cuối)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Đóng băng ResNet (tùy chọn, để False nếu muốn train cả ResNet)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Temporal: LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)

    def forward(self, x):
        # Input: (Batch, Seq, C, H, W)
        b, s, c, h, w = x.size()
        x = x.view(b*s, c, h, w)
        
        # CNN Feature Extract
        features = self.backbone(x) # (B*S, 512, 1, 1)
        features = features.view(b, s, -1) # (B, S, 512)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Trả về hidden state cuối cùng (B, 256)
        return h_n[-1]

# 2. MODEL CHO GIAI ĐOẠN 1 (Contrastive)
# Encoder + Projection Head
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = BaseEncoder()
        # Projection head: Chỉ dùng để so sánh, vứt đi sau khi pre-train
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM)
        )

    def forward_one(self, x):
        feat = self.encoder(x)
        return self.projection(feat)

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)

# 3. MODEL CHO GIAI ĐOẠN 2 (Classification)
# Encoder (đã train) + Classifier Head
class ClassifierNetwork(nn.Module):
    def __init__(self, encoder_weights_path=None):
        super(ClassifierNetwork, self).__init__()
        self.encoder = BaseEncoder()
        
        # Load weights từ giai đoạn 1 nếu có
        if encoder_weights_path:
            print(f"Loading encoder weights from {encoder_weights_path}...")
            # Load state dict
            checkpoint = torch.load(encoder_weights_path)
            # Lọc bỏ phần 'projection' nếu lỡ lưu thừa, chỉ lấy phần 'encoder.'
            # Tuy nhiên code stage 1 bên dưới sẽ chỉ lưu encoder nên load thẳng:
            self.encoder.load_state_dict(checkpoint)
            print("-> Weights loaded successfully!")
        
        # Classifier Head mới: Nhận input từ LSTM (256) chứ không phải Projection (128)
        self.head = nn.Linear(256, 1) # Binary classification

    def forward(self, x):
        feat = self.encoder(x) # Output (B, 256)
        return self.head(feat)