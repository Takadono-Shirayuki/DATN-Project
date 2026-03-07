"""
Gait Recognition Module - Nhận dạng dáng đi từ video stream
Sử dụng ResNetEmbedder để tạo embedding 128-dim và OpenSetGaitMatcher để nhận dạng
"""
import os
import numpy as np
import torch
from collections import deque
from PIL import Image
import cv2

from open_set.encoder_resnet import ResNetEmbedder
from open_set.gei import make_gei_from_frames
from open_set.open_set_matcher import OpenSetGaitMatcher


class GaitRecognizer:
    """
    Nhận dạng dáng đi real-time từ video stream
    Sử dụng OpenSetGaitMatcher với database.json
    """
    def __init__(self, 
                 model_path='open_set/encoder_resnet.pth',
                 database_path='database.json',
                 buffer_size=30,
                 embedding_dim=128,
                 cooldown_frames=60):
        """
        Args:
            model_path: Đường dẫn đến model encoder
            database_path: Đường dẫn đến database.json (OpenSetGaitMatcher format)
            buffer_size: Số frame để tạo GEI (30 frames ~ 1 giây)
            embedding_dim: Kích thước embedding
            cooldown_frames: Số frames delay sau khi nhận dạng thành công (60 frames ~ 2 giây)
        """
        self.buffer_size = buffer_size
        self.cooldown_frames = cooldown_frames
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoder model
        print(f"Loading gait encoder from {model_path}...")
        self.model = ResNetEmbedder(embedding_dim=embedding_dim, pretrained=False, grayscale=True)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Gait encoder loaded on {self.device}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load OpenSetGaitMatcher
        self.matcher = None
        if os.path.exists(database_path):
            self.matcher = OpenSetGaitMatcher(metric='cosine', filename=database_path)
            print(f"✓ OpenSetGaitMatcher loaded from {database_path}")
        else:
            print(f"⚠ Database not found: {database_path}")
            print("  Run: python run_open_set.py gait/embeddings.json --output_db database.json")
        
        # Frame buffers per person
        self.person_buffers = {}  # {person_id: deque of frames}
        self.person_labels = {}   # {person_id: recognized_name}
        self.last_recognition_frame = {}  # {person_id: frame_count}
        self.frame_count = {}  # {person_id: total_frames_seen}
        self._gei_saved = False  # DEBUG: save GEI once
    
    def update(self, person_id, frame, is_walking, mask=None, keypoints=None):
        """
        Cập nhật frame cho một người và thực hiện nhận dạng nếu đủ frames
        
        Args:
            person_id: ID của người (ví dụ: 'Person_1')
            frame: Frame ảnh của người (H, W, 3) hoặc (H, W) - không còn dùng cho GEI
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
        
        # Tăng frame counter
        self.frame_count[person_id] += 1
        
        # Chỉ tích lũy frames khi đang walking
        if is_walking and mask is not None:
            # Mask đã được resize về 224x224 với aspect ratio preservation trong camera_module
            # Chỉ cần đảm bảo format đúng
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            self.person_buffers[person_id].append(mask)
            
            # Kiểm tra cooldown
            frames_since_last = self.frame_count[person_id] - self.last_recognition_frame[person_id]
            can_recognize = frames_since_last >= self.cooldown_frames
            
            # Khi đủ frames VÀ hết cooldown → nhận dạng
            if len(self.person_buffers[person_id]) == self.buffer_size and can_recognize:
                recognized_name = self._recognize(person_id)
                if recognized_name:
                    self.person_labels[person_id] = recognized_name
                    self.last_recognition_frame[person_id] = self.frame_count[person_id]
                    print(f"✓ Recognized {person_id} as {recognized_name}")
                self.person_buffers[person_id].clear()
        else:
            # Không walking → clear buffer, giữ label
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

            # DEBUG: lưu GEI đầu tiên tạo được để kiểm tra
            if not self._gei_saved:
                debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_gei.png')
                cv2.imwrite(debug_path, (gei * 255).astype(np.uint8))
                print(f"[DEBUG] GEI saved to {debug_path}")
                self._gei_saved = True

            gei_tensor = torch.from_numpy(gei).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            gei_tensor = (gei_tensor - 0.5) / 0.5  # Normalize
            gei_tensor = gei_tensor.to(self.device)
            
            with torch.no_grad():
                embedding = self.model(gei_tensor).cpu().numpy().flatten()
            
            # Dùng OpenSetGaitMatcher để predict
            result = self.matcher.predict(embedding)
            
            # result = {'user_id': ..., 'distance': ..., 'threshold': ..., 'is_known': ...}
            print(f"\n=== Gait Recognition ({person_id}) ===")
            print(f"  User ID: {result['user_id']}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Threshold: {result.get('threshold', 'N/A')}")
            print(f"  Is Known: {result['is_known']}")
            
            if result['is_known']:
                user_id = result['user_id']
                # Convert user_id to string name if needed
                return str(user_id)
            else:
                print(f"  ✗ REJECTED: Unknown person")
                return None
                
        except Exception as e:
            print(f"Error in gait recognition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reset(self, person_id=None):
        """Reset buffer và label cho một người hoặc tất cả"""
        if person_id is None:
            self.person_buffers.clear()
            self.person_labels.clear()
            self.last_recognition_frame.clear()
            self.frame_count.clear()
        else:
            if person_id in self.person_buffers:
                self.person_buffers[person_id].clear()
            if person_id in self.person_labels:
                del self.person_labels[person_id]
            if person_id in self.last_recognition_frame:
                del self.last_recognition_frame[person_id]
            if person_id in self.frame_count:
                del self.frame_count[person_id]
    
    def get_label(self, person_id):
        """Lấy label đã nhận dạng cho một người"""
        return self.person_labels.get(person_id)


if __name__ == '__main__':
    # Test: Tạo database.json từ notebook
    print("To create database.json:")
    print("1. Run train_gei_encoder.ipynb to train model and extract embeddings")
    print("2. Cell 29 in notebook will save embeddings.json")
    print("3. Cell 30 in notebook will create database.json using OpenSetGaitMatcher")
    print("\nOr run manually:")
    print("  python run_open_set.py gait/embeddings.json --output_db database.json")
