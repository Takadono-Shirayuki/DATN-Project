import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from common import ClassifierNetwork, SEQ_LEN
import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = str(SCRIPT_DIR.parent / "dataset_processed")
BATCH_SIZE = 16
EPOCHS = 10
LR_ENCODER = 1e-4
LR_HEAD = 1e-3
ENCODER_PATH = "encoder_pretrained.pth"
FINAL_MODEL_PATH = "contrastive_model.pth"
HISTORY_FILE = "training_history.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATASET ---
class ActionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = [] 
        for label_name, label_idx in {'non_walking': 0, 'walking': 1}.items():
            class_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(class_dir): continue
            for vid in os.listdir(class_dir):
                if vid.startswith('.'): continue
                self.samples.append((os.path.join(class_dir, vid), label_idx))

    def __len__(self): return len(self.samples)

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

def run_stage2():
    print("=== STAGE 2: FINE-TUNING CLASSIFIER ===")
    
    history = {
        'epoch': [], 
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': []
    }
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = ActionDataset(DATA_DIR, 'train', transform)
    val_ds = ActionDataset(DATA_DIR, 'val', transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    if os.path.exists(ENCODER_PATH):
        print(f"Loading Pretrained Encoder: {ENCODER_PATH}")
        model = ClassifierNetwork(encoder_weights_path=ENCODER_PATH).to(DEVICE)
    else:
        model = ClassifierNetwork().to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
            {'params': model.head.parameters(), 'lr': LR_HEAD}
        ])

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_corr = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            # Tính chỉ số
            train_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            train_corr += (preds == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=train_corr/train_total)

        # Tính trung bình Train
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_corr / train_total

        # --- VAL ---
        model.eval()
        val_loss = 0
        val_corr = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                out = model(imgs)
                loss = criterion(out, labels)
                
                # Tính chỉ số
                val_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                val_corr += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        # Tính trung bình Val
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_corr / val_total
        
        print(f" -> Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc*100:.2f}%")
        print(f" -> Val Loss:   {avg_val_loss:.4f} | Acc: {avg_val_acc*100:.2f}%")

        # --- SAVE HISTORY ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)

    # Lưu model cuối cùng
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Done! Model saved to {FINAL_MODEL_PATH}")
    print(f"History saved to {HISTORY_FILE}")

if __name__ == "__main__":
    run_stage2()