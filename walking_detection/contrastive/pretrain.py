import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import numpy as np

# Import từ common
try:
    from common import SiameseNetwork, SEQ_LEN
except ImportError:
    print("Lỗi: Không tìm thấy file common.py")
    exit(1)

# --- CONFIG ---
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = str(SCRIPT_DIR.parent / "dataset_processed")
BATCH_SIZE = 16 
EPOCHS = 5
LR = 1e-4
ENCODER_SAVE_PATH = "encoder_pretrained.pth"
HISTORY_FILE = "stage1_history.json" # <--- [MỚI] Tên file lịch sử
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps" # Hỗ trợ Mac M1/M2

# --- DATASET (Siamese Pairs) ---
class SiameseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.data_dict = {0: [], 1: []} # 0: non_walking, 1: walking
        
        # Load danh sách video
        for label_name, label_idx in {'non_walking': 0, 'walking': 1}.items():
            class_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(class_dir): continue
            for vid in os.listdir(class_dir):
                if vid.startswith('.'): continue
                self.data_dict[label_idx].append(os.path.join(class_dir, vid))
        
        self.all_videos = [(p, 0) for p in self.data_dict[0]] + [(p, 1) for p in self.data_dict[1]]

    def __len__(self): return len(self.all_videos)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        if len(files) < SEQ_LEN: return torch.zeros(SEQ_LEN, 3, 224, 224), label
        files = files[:SEQ_LEN]
        frames = []
        for file in files:
            img = Image.open(os.path.join(video_path, file)).convert('RGB')
            if self.transform: img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), label

# --- LOSS FUNCTION ---
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = torch.mean(label * torch.pow(dist, 2) + 
                          (1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

# --- MAIN ---
def run_stage1():
    print(f"=== STAGE 1: CONTRASTIVE PRE-TRAINING (Device: {DEVICE}) ===")
    
    # [MỚI] Khởi tạo lịch sử
    history = {'epoch': [], 'loss': []}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = SiameseDataset(DATA_DIR, 'train', transform=transform)
    
    # 2 data loader, 8 walking, 8 non_walking
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SiameseNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = ContrastiveLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img1, img2, target in loop:
            img1, img2, target = img1.to(DEVICE), img2.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            o1, o2 = model(img1, img2)
            loss = criterion(o1, o2, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # [MỚI] Tính trung bình loss và lưu vào JSON
        avg_loss = total_loss / len(loader)
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        
        print(f" -> Avg Loss: {avg_loss:.4f}")
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
            
    # Save Encoder
    print(f"Saving encoder to {ENCODER_SAVE_PATH}...")
    torch.save(model.encoder.state_dict(), ENCODER_SAVE_PATH)
    print("Done Stage 1!")

if __name__ == "__main__":
    run_stage1()