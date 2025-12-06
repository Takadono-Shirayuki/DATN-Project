import json

# Create train_action_classifier notebook structure  
classifier_nb = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Action Recognition Training - Classification Model\n\nThis notebook trains a simple LSTM classifier to recognize walking vs non-walking actions from keypoint sequences.']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 1. Import Libraries']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['import os\nimport json\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import Dataset, DataLoader\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nimport matplotlib.pyplot as plt\nfrom tqdm import tqdm']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 2. Configuration']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['DATASET_DIR = "dataset"\nSEQUENCE_LENGTH = 30\nSTRIDE = 15\nBATCH_SIZE = 32\nNUM_EPOCHS = 50\nLEARNING_RATE = 0.001\nHIDDEN_SIZE = 128\nNUM_LAYERS = 2\nDROPOUT = 0.3\n\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\nprint(f"Using device: {device}")']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 3. Define Dataset Class\n\nSee `train_action_classifier.py` for full implementation']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 4. Define Model Architecture\n\nSee `train_action_classifier.py` for full ActionClassifier model']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 5. Load and Prepare Dataset']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 6. Train Model']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 7. Evaluate and Visualize Results']},
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.8.0'}},
    'nbformat': 4,
    'nbformat_minor': 4
}

# Create train_action_contrastive notebook structure
contrastive_nb = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Action Recognition Training - Contrastive Learning Model\n\nThis notebook uses contrastive learning (SimCLR-style) for learning action representations from keypoint sequences.']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 1. Import Libraries']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['import os\nimport json\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torch.nn.functional as F\nfrom torch.utils.data import Dataset, DataLoader\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, classification_report\nimport matplotlib.pyplot as plt\nfrom tqdm import tqdm\nimport random']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 2. Configuration']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': ['DATASET_DIR = "dataset"\nSEQUENCE_LENGTH = 30\nSTRIDE = 15\nBATCH_SIZE = 32\nNUM_EPOCHS = 100\nLEARNING_RATE = 0.001\nTEMPERATURE = 0.5\nHIDDEN_SIZE = 128\nNUM_LAYERS = 2\nDROPOUT = 0.3\n\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\nprint(f"Using device: {device}")']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 3. Define Contrastive Dataset Class\n\nSee `train_action_contrastive.py` for full implementation with data augmentation']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 4. Define Model Architecture\n\nSee `train_action_contrastive.py` for LSTM Encoder and Contrastive Model']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 5. Define Contrastive Loss (NTXent)\n\nNormalized Temperature-scaled Cross Entropy Loss from SimCLR']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 6. Load and Prepare Dataset']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 7. Train Model (Two-Phase)\n\nPhase 1: Contrastive pre-training (60% of epochs)\nPhase 2: Fine-tuning with classification (40% of epochs)']},
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 8. Evaluate and Visualize Results']},
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.8.0'}},
    'nbformat': 4,
    'nbformat_minor': 4
}

# Save notebooks
with open('train_action_classifier.ipynb', 'w') as f:
    json.dump(classifier_nb, f, indent=2)

with open('train_action_contrastive.ipynb', 'w') as f:
    json.dump(contrastive_nb, f, indent=2)

print('✓ Notebooks created successfully!')
