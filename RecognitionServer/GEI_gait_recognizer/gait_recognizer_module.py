"""
Gait Recognition Module - Nhận dạng dáng đi từ video stream
Sử dụng ResNetEmbedder để tạo embedding 128-dim và OpenSetGaitMatcher để nhận dạng
"""
import os
import numpy as np
import torch
from collections import deque

from .encoder_resnet import ResNetEmbedder
from .gei import make_gei_from_frames
from .open_set_matcher import OpenSetGaitMatcher


class GaitRecognizer:
    """
    Nhận dạng dáng đi real-time từ video stream
    Sử dụng OpenSetGaitMatcher với database.json
    """
    def __init__(self, 
                 model_path=None,
                 database_path=None,
                 buffer_size=30,
                 embedding_dim=128,
                 cooldown_frames=60):
        """
        Args:
            model_path: Đường dẫn đến model encoder (mặc định: encoder_resnet.pth cùng thư mục)
            database_path: Đường dẫn đến database.json (mặc định: database.json cùng thư mục)
            buffer_size: Số frame để tạo GEI (30 frames ~ 1 giây)
            embedding_dim: Kích thước embedding
            cooldown_frames: Số frames delay sau khi nhận dạng thành công (60 frames ~ 2 giây)
        """
        _dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(_dir, 'encoder_resnet.pth')
        if database_path is None:
            database_path = os.path.join(_dir, 'database.json')
        self.buffer_size = buffer_size
        self.cooldown_frames = cooldown_frames
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoder model
        self.model = ResNetEmbedder(embedding_dim=embedding_dim, pretrained=False, grayscale=True)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load OpenSetGaitMatcher
        self.matcher = None
        if os.path.exists(database_path):
            self.matcher = OpenSetGaitMatcher(metric='cosine', filename=database_path)
        else:
            print(f"⚠ Database not found: {database_path}")
            print("  Run: python run_open_set.py gait/embeddings.json --output_db database.json")
        
        # Frame buffers per person
        self.person_buffers = {}  # {person_id: deque of frames}
        self.person_labels = {}   # {person_id: recognized_name}
        self.last_recognition_frame = {}  # {person_id: frame_count}
        self.frame_count = {}  # {person_id: total_frames_seen}
    
    def push(self, person_id, is_walking, mask=None, keypoints=None):
        """
        Cập nhật frame cho một người và thực hiện nhận dạng nếu đủ frames
        
        Args:
            person_id: ID của người (ví dụ: 'Person_1')
            is_walking: Người đang đi bộ hay không
            mask: Binary mask/silhouette của người (H, W) - CẦN CHO GEI
            keypoints: Keypoints (optional)
        
        Returns:
            recognized_name: Tên người được nhận dạng (hoặc None nếu chưa nhận dạng)
        """
        if self.matcher is None:
            return None
        
        # Khởi tạo buffer cho person mới
        if person_id not in self.person_buffers:
            self.person_buffers[person_id] = deque(maxlen=self.buffer_size)
            self.person_labels[person_id] = None
            self.last_recognition_frame[person_id] = -self.cooldown_frames
            self.frame_count[person_id] = 0
        
        self.frame_count[person_id] += 1
        
        if is_walking and mask is not None:
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            self.person_buffers[person_id].append(mask)
            
            frames_since_last = self.frame_count[person_id] - self.last_recognition_frame[person_id]
            can_recognize = frames_since_last >= self.cooldown_frames
            
            if len(self.person_buffers[person_id]) == self.buffer_size and can_recognize:
                recognized_name = self._recognize(person_id)
                if recognized_name:
                    self.person_labels[person_id] = recognized_name
                    self.last_recognition_frame[person_id] = self.frame_count[person_id]
                self.person_buffers[person_id].clear()
        else:
            self.person_buffers[person_id].clear()
        
        return self.person_labels.get(person_id)
    
    def _recognize(self, person_id):
        """
        Nhận dạng người từ buffer frames sử dụng OpenSetGaitMatcher
        
        Returns:
            recognized_name: Tên người hoặc None nếu không nhận dạng được (unknown/rejected)
        """
        try:
            # Tạo GEI từ frames
            frames = list(self.person_buffers[person_id])
            if len(frames) < self.buffer_size:
                return None
            
            gei = make_gei_from_frames(frames, size=(224, 224))

            gei_tensor = torch.from_numpy(gei).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            gei_tensor = (gei_tensor - 0.5) / 0.5  # Normalize
            gei_tensor = gei_tensor.to(self.device)
            
            with torch.no_grad():
                embedding = self.model(gei_tensor).cpu().numpy().flatten()
            
            # Dùng OpenSetGaitMatcher để predict
            result = self.matcher.predict(embedding)
            
            if result['is_known']:
                return str(result['user_id'])
            return None
                
        except Exception as e:
            print(f"Error in gait recognition: {e}")
            import traceback
            traceback.print_exc()
            return None