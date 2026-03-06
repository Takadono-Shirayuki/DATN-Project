"""
Action Classifier Module
Bi-directional LSTM model for walking/non-walking action recognition
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque


class ActionClassifier(nn.Module):
    """Bi-directional LSTM classifier for action recognition"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(ActionClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
            
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class ActionRecognizer:
    """
    Action recognition pipeline for real-time inference
    
    Features:
    - Maintains sequence buffer for each person ID
    - Extracts 12 body keypoints (arms, legs, hips)
    - Quality filtering (min confidence, min valid keypoints)
    - Real-time prediction with temporal smoothing
    """
    
    # 12 body keypoints: shoulders(5,6), arms(7,8,9,10), hips(11,12), legs(13,14,15,16)
    BODY_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    def __init__(self, model_path='action_classifier_128_2.pth', 
                 sequence_length=30, 
                 min_confidence=0.3,
                 min_valid_keypoints=8,
                 device=None):
        """
        Initialize action recognizer
        
        Args:
            model_path: Path to trained model weights (.pth file)
            sequence_length: Number of frames required for prediction
            min_confidence: Minimum keypoint confidence threshold
            min_valid_keypoints: Minimum number of valid keypoints per frame
            device: torch device (cuda/cpu), auto-detect if None
        """
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.min_valid_keypoints = min_valid_keypoints
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Sequence buffers for each person: {person_id: deque of frames}
        self.sequence_buffers = {}
        
        # Prediction history for temporal smoothing: {person_id: deque of predictions}
        self.prediction_history = {}
        self.history_size = 5  # Number of predictions to average
        
        # Load model
        input_size = len(self.BODY_KEYPOINTS) * 3  # 12 keypoints * (x, y, confidence)
        self.model = ActionClassifier(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=2,
            dropout=0.3
        ).to(self.device)
        
        # Load trained weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Loaded action classifier from {model_path}")
            print(f"   Device: {self.device}")
            print(f"   Input size: {input_size} (12 keypoints × 3)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Class names
        self.class_names = ['Non-Walking', 'Walking']
    
    def _is_frame_valid(self, keypoints):
        """
        Check if frame has enough high-quality keypoints
        
        Args:
            keypoints: List of 17 keypoints, each [x, y, confidence]
        Returns:
            bool: True if frame is valid
        """
        if not isinstance(keypoints, (list, np.ndarray)) or len(keypoints) < 17:
            return False
        
        valid_count = 0
        for kp_idx in self.BODY_KEYPOINTS:
            if kp_idx < len(keypoints):
                keypoint = keypoints[kp_idx]
                if len(keypoint) >= 3:
                    confidence = keypoint[2]
                    if confidence >= self.min_confidence:
                        valid_count += 1
        
        return valid_count >= self.min_valid_keypoints
    
    def _extract_features(self, keypoints, bbox):
        """
        Extract body keypoints features from frame
        Transform keypoints EXACTLY like training data preprocessing:
        1. Crop square region centered on bbox with size=max(bbox_w, bbox_h)
        2. Transform keypoints to cropped coordinates
        3. Resize to 224x224 (keypoints scale accordingly)
        4. Normalize to [0,1]
        
        Args:
            keypoints: List of 17 keypoints, each [x, y, confidence] in original frame coords
            bbox: [x1, y1, x2, y2] bounding box
        Returns:
            np.array: Flattened features [36] (12 keypoints × 3)
        """
        features = []
        
        # Extract bbox info
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Calculate crop region (same as training data)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        crop_size = max(bbox_w, bbox_h)  # Square crop
        
        crop_x1 = center_x - crop_size / 2
        crop_y1 = center_y - crop_size / 2
        
        # Scale factor: crop_size -> 224
        scale = 224.0 / crop_size if crop_size > 0 else 1.0
        
        for kp_idx in self.BODY_KEYPOINTS:
            if kp_idx < len(keypoints):
                keypoint = keypoints[kp_idx]
                if len(keypoint) >= 3:
                    x, y, conf = keypoint[0], keypoint[1], keypoint[2]
                    
                    # Transform to crop coordinates
                    x_crop = (x - crop_x1) * scale
                    y_crop = (y - crop_y1) * scale
                    
                    # Clip to 224x224
                    x_clipped = np.clip(x_crop, 0, 224)
                    y_clipped = np.clip(y_crop, 0, 224)
                    
                    # Normalize to [0, 1] like training
                    x_norm = x_clipped / 224.0
                    y_norm = y_clipped / 224.0
                    conf_norm = np.clip(conf, 0, 1)
                    
                    features.extend([x_norm, y_norm, conf_norm])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def update(self, person_id, keypoints, bbox):
        """
        Update sequence buffer with new frame keypoints
        
        Args:
            person_id: Unique identifier for the person
            keypoints: List of 17 keypoints from YOLOv8-pose
            bbox: [x1, y1, x2, y2] bounding box for this person
        """
        # Initialize buffer if new person
        if person_id not in self.sequence_buffers:
            self.sequence_buffers[person_id] = deque(maxlen=self.sequence_length)
            self.prediction_history[person_id] = deque(maxlen=self.history_size)
        
        # Check frame quality
        if self._is_frame_valid(keypoints):
            features = self._extract_features(keypoints, bbox)
            self.sequence_buffers[person_id].append(features)
    
    def predict(self, person_id, use_smoothing=True):
        """
        Predict action for a person
        
        Args:
            person_id: Unique identifier for the person
            use_smoothing: Whether to use temporal smoothing (averaging last N predictions)
        
        Returns:
            dict: {
                'action': str,           # 'Walking' or 'Non-Walking'
                'confidence': float,     # Probability [0-1]
                'ready': bool,           # True if enough frames collected
                'buffer_size': int       # Current number of frames in buffer
            }
        """
        # Check if person exists and has enough frames
        if person_id not in self.sequence_buffers:
            return {
                'action': 'Unknown',
                'confidence': 0.0,
                'ready': False,
                'buffer_size': 0
            }
        
        buffer = self.sequence_buffers[person_id]
        buffer_size = len(buffer)
        
        if buffer_size < self.sequence_length:
            return {
                'action': 'Collecting...',
                'confidence': 0.0,
                'ready': False,
                'buffer_size': buffer_size
            }
        
        # Prepare input sequence
        sequence = np.array(list(buffer), dtype=np.float32)  # Shape: (30, 36)
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Shape: (1, 30, 36)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted.item()
        
        # Temporal smoothing
        if use_smoothing:
            self.prediction_history[person_id].append(predicted_class)
            
            # Average predictions over history
            if len(self.prediction_history[person_id]) >= 3:
                avg_prediction = np.mean(list(self.prediction_history[person_id]))
                predicted_class = 1 if avg_prediction >= 0.5 else 0
        
        action = self.class_names[predicted_class]
        
        return {
            'action': action,
            'confidence': confidence,
            'ready': True,
            'buffer_size': buffer_size
        }
    
    def reset(self, person_id=None):
        """
        Reset sequence buffer for a person (or all if person_id is None)
        
        Args:
            person_id: Person to reset, or None to reset all
        """
        if person_id is None:
            self.sequence_buffers.clear()
            self.prediction_history.clear()
        else:
            if person_id in self.sequence_buffers:
                del self.sequence_buffers[person_id]
            if person_id in self.prediction_history:
                del self.prediction_history[person_id]
    
    def get_active_persons(self):
        """
        Get list of active person IDs
        
        Returns:
            list: List of person IDs with active buffers
        """
        return list(self.sequence_buffers.keys())
    
    def cleanup_inactive(self, active_ids):
        """
        Remove buffers for persons no longer in frame
        
        Args:
            active_ids: List of currently active person IDs
        """
        inactive_ids = set(self.sequence_buffers.keys()) - set(active_ids)
        for person_id in inactive_ids:
            self.reset(person_id)