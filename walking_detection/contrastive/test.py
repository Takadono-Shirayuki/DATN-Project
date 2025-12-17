import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# --- IMPORT FROM COMMON (IMPORTANT) ---
try:
    from common import ClassifierNetwork, SEQ_LEN
except ImportError:
    print("Error: Could not find common.py. Make sure it is in the same directory.")
    exit(1)

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = str(SCRIPT_DIR.parent / "dataset_processed")
BATCH_SIZE = 16
MODEL_PATH = "contrastive_model.pth"
OUTPUT_IMAGE = "test_result_contrastive.png"

# DEVICE
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
DEVICE = get_device()

# --- DATASET ---
# We redefine Dataset here to make test.py standalone, 
# ensuring it matches the logic used in Stage 2.
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

# --- TEST FUNCTION ---
def test():
    print(f"Testing Model on {DEVICE}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("Please run stage2_classify.py first to generate the final model.")
        return

    model = ClassifierNetwork().to(DEVICE)
    
    print(f"Loading weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading keys: {e}")
        print("Hint: Did you change the architecture in common.py after training?")
        return

    model.eval()

    # 3. Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    split_to_use = 'test'
    if not os.path.exists(os.path.join(DATA_DIR, 'test')):
        print("Warning: 'test' folder not found, using 'val' for testing.")
        split_to_use = 'val'
        
    dataset = ActionDataset(DATA_DIR, split=split_to_use, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(dataset) == 0:
        print(f"Error: No data found in {DATA_DIR}/{split_to_use}")
        return

    y_true, y_pred = [], []
    
    print(f"Running Inference on {len(dataset)} samples...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            
            preds = (torch.sigmoid(out) > 0.5).long().cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten()

    print("\n" + "="*30)
    print("       TEST REPORT       ")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=['Non-Walking', 'Walking']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Walking', 'Walking'], 
                yticklabels=['Non-Walking', 'Walking'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
    print(f"✅ Confusion Matrix saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    test()