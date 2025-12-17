import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# --- CẤU HÌNH ---
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = str(SCRIPT_DIR.parent / "dataset_processed")
BATCH_SIZE = 8
SEQ_LEN = 32
MODEL_PATH = "action_recognition_model.pth"
OUTPUT_IMAGE_FILE = "test_result.png"

# 1. SETUP DEVICE
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
DEVICE = get_device()

# 2. MODEL ARCHITECTURE
class CRNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CRNN, self).__init__()
        resnet = models.resnet18(weights=None) # Không cần load pretrain weights vì sẽ load từ file .pth
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out

# 3. DATASET CLASS
class ActionDataset(Dataset):
    def __init__(self, root_dir, split='test', transform=None):
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

    def __len__(self):
        return len(self.samples)

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

# 4. TESTING FUNCTION
def test():
    print(f"Testing on device: {DEVICE}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ActionDataset(DATA_DIR, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = CRNN(num_classes=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully!")
    else:
        print("Error: Model file not found!")
        return

    model.eval()
    
    y_true = []
    y_pred = []

    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten() # Flatten để khớp dimension

    # REPORT
    print("\n" + "="*30)
    print("TEST REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=['Non-Walking', 'Walking']))

    # CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Walking', 'Walking'], yticklabels=['Non-Walking', 'Walking'])
    plt.ylabel('Thực tế (Ground Truth)')
    plt.xlabel('Dự đoán (Prediction)')
    plt.title('Confusion Matrix')

    plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight') 
    print(f"\n-> Đã lưu ảnh kết quả vào: {OUTPUT_IMAGE_FILE}")

    plt.show()

if __name__ == "__main__":
    test()