import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
# Get dataset_processed from parent directory relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = str(SCRIPT_DIR.parent / "dataset_processed")
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
SEQ_LEN = 32
HISTORY_FILE = "training_history.json" # New JSON file
MODEL_FILE = "action_recognition_model.pth"

# --- DEVICE ---
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
DEVICE = get_device()

# --- DATASET & MODEL (Same as before) ---
class ActionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = [] 
        self.class_map = {'non_walking': 0, 'walking': 1}
        for label_name, label_idx in self.class_map.items():
            class_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(class_dir): continue
            for video_folder in os.listdir(class_dir):
                if video_folder.startswith('.'): continue
                self.samples.append((os.path.join(class_dir, video_folder), label_idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        if len(files) < SEQ_LEN: return torch.zeros(SEQ_LEN, 3, 224, 224), label
        frames = [Image.open(os.path.join(video_path, f)).convert('RGB') for f in files[:SEQ_LEN]]
        if self.transform: frames = [self.transform(img) for img in frames]
        return torch.stack(frames), label

class CRNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CRNN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        for param in self.backbone.parameters(): param.requires_grad = False
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        return self.fc(h_n[-1])

# --- TRAIN FUNCTION ---
def train():
    # 1. Initialize History (Create new or overwrite)
    history_data = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []
    }
    # Reset JSON file at start
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Dataset...")
    train_dataset = ActionDataset(DATA_DIR, split='train', transform=transform)
    val_dataset = ActionDataset(DATA_DIR, split='val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CRNN(num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Start training on {DEVICE}...")

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        # VAL
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate Stats
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        print(f" -> Train Loss: {avg_train_loss:.4f} | Train Acc: {100 * avg_train_acc:.2f}%")
        print(f" -> Val Loss:   {avg_val_loss:.4f} | Val Acc:   {100 * avg_val_acc:.2f}%")

        # --- SAVE TO JSON ---
        # Update local dictionary
        history_data['epoch'].append(epoch + 1)
        history_data['train_loss'].append(avg_train_loss)
        history_data['val_loss'].append(avg_val_loss)
        history_data['train_acc'].append(avg_train_acc)
        history_data['val_acc'].append(avg_val_acc)

        # Write to disk
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        # Save Model every epoch (optional, safety)
        torch.save(model.state_dict(), MODEL_FILE)

    print(f"Training finished. History saved to {HISTORY_FILE}")

if __name__ == "__main__":
    train()
