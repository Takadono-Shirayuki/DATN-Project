import numpy as np

# PHẦN 1: Mô phỏng các thuật toán tính khoảng cách và ngưỡng Open-set từ tài liệu

class OpenSetGaitMatcher:
    def __init__(self, metric='euclidean', k_factor=3.0):
        """
        Khởi tạo bộ so khớp.
        :param metric: 'euclidean' hoặc 'cosine'
        :param k_factor: Hệ số k để tính ngưỡng (Mean + k*Sigma)
        """
        self.metric = metric
        self.k_factor = k_factor
        self.database = {} # Nơi lưu Centroid và Threshold của từng người

    def _calculate_distance(self, vec1, vec2):
        """Hàm tính khoảng cách giữa 2 vector"""
        if self.metric == 'euclidean':
            # Khoảng cách Euclid (L2 norm)
            return np.linalg.norm(vec1 - vec2)
        
        elif self.metric == 'cosine':
            # Khoảng cách Cosine = 1 - Cosine Similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0: return 1.0
            cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return 1 - cosine_similarity
        
        else:
            raise ValueError("Metric không hợp lệ. Chọn 'euclidean' hoặc 'cosine'")

    def enroll_user(self, user_id, vectors_list):
        """Bước Đăng ký: Tính Centroid và Ngưỡng riêng cho người dùng."""
        vectors = np.array(vectors_list)
        
        # 1. Tính Centroid (Vector trung bình)
        centroid = np.mean(vectors, axis=0)
        
        # 2. Tính phân bố khoảng cách nội lớp (Intra-class distribution)
        distances = []
        for vec in vectors:
            d = self._calculate_distance(vec, centroid)
            distances.append(d)
            
        # 3. Tính Mean (mu) và Std Dev (sigma)
        mu = np.mean(distances)
        sigma = np.std(distances)
        
        # 4. Tính Ngưỡng Adaptive: Threshold = mu + k * sigma
        threshold = mu + self.k_factor * sigma
        
        # Lưu vào database giả lập
        self.database[user_id] = {
            'centroid': centroid,
            'threshold': threshold,
            'stats': {'mu': mu, 'sigma': sigma} # Lưu để debug xem chơi
        }
        print(f"✅ Đã đăng ký {user_id}: Threshold = {threshold:.4f} (Mu={mu:.4f}, Sigma={sigma:.4f})")

    def identify(self, probe_vector):
        """
        Bước Nhận diện: So khớp vector lạ với database.
        Trả về: (User_ID, Khoảng cách) hoặc ("Unknown", Khoảng cách gần nhất)
        (Source [57-64])
        """
        min_dist = float('inf')
        identified_user = "Unknown"
        
        print(f"\n--- Đang check Probe Vector ---")
        
        for user_id, data in self.database.items():
            centroid = data['centroid']
            threshold = data['threshold']
            
            # Tính khoảng cách từ Probe đến Centroid từng người
            dist = self._calculate_distance(probe_vector, centroid)
            
            print(f" > So với {user_id}: Dist = {dist:.4f} / Thresh = {threshold:.4f}", end="")
            
            # Kiểm tra xem có nằm trong ngưỡng không [Source: 63]
            if dist <= threshold:
                print(f" -> KHỚP ✅")
                # Nếu khớp nhiều người (hiếm), lấy người có khoảng cách nhỏ nhất
                if dist < min_dist:
                    min_dist = dist
                    identified_user = user_id
            else:
                print(f" -> LỆCH ❌")

        return identified_user

# ==============================================================================
# PHẦN 2: GIẢ LẬP DỮ LIỆU & CHẠY THỬ (STAGE 1 & STAGE 2)
# ==============================================================================

def run_simulation():
    # Cấu hình giả lập
    VECTOR_DIM = 128  # Giả sử vector embedding có 128 chiều [Source: 48]
    NUM_SAMPLES = 20  # Mỗi người có 20 mẫu để đăng ký
    
    # --- Bước 1: Tạo dữ liệu giả (Mock Data Generation) ---
    print("=== BƯỚC 1: GIẢ LẬP DỮ LIỆU EMBEDDING (Từ Model AI) ===")
    
    # Hàm sinh dữ liệu cho một người (gồm 1 vector gốc + nhiễu ngẫu nhiên)
    def generate_person_data(base_seed):
        np.random.seed(base_seed)
        base_vector = np.random.rand(VECTOR_DIM) # Vector gốc "dáng đi chuẩn"
        samples = []
        for _ in range(NUM_SAMPLES):
            # Thêm nhiễu (noise) để giả lập việc đi mỗi lần mỗi khác một tí
            noise = np.random.normal(0, 0.05, VECTOR_DIM) 
            samples.append(base_vector + noise)
        return samples

    # Tạo dữ liệu cho 2 nhân viên: Alice và Bob
    data_alice = generate_person_data(100) # Alice
    data_bob = generate_person_data(200)   # Bob
    
    # Tạo dữ liệu cho người lạ (Eve) - không đăng ký
    np.random.seed(999)
    vector_eve_unknown = np.random.rand(VECTOR_DIM) # Dáng đi hoàn toàn khác

    print("Đã sinh xong dữ liệu cho Alice và Bob.")

    # --- Bước 2: Chạy thử Logic của bạn ---
    print("\n=== BƯỚC 2: CHẠY THUẬT TOÁN TÍNH NGƯỠNG & OPEN-SET ===")

    # KHỞI TẠO HỆ THỐNG
    # Bạn có thể đổi 'euclidean' thành 'cosine' ở đây để test version 2
    my_system = OpenSetGaitMatcher(metric='euclidean', k_factor=3.0) 
    
    # A. ĐĂNG KÝ (ENROLLMENT)
    print("\n--- Bắt đầu đăng ký ---")
    my_system.enroll_user("Alice", data_alice)
    my_system.enroll_user("Bob", data_bob)
    
    # B. KIỂM TRA (INFERENCE)
    
    # Case 1: Alice đi vào (Lấy 1 mẫu mới của Alice tạo từ cùng seed nhưng nhiễu khác)
    print("\n[TEST 1] Người thật (Alice) đi vào:")
    np.random.seed(100)
    probe_alice = np.random.rand(VECTOR_DIM) + np.random.normal(0, 0.05, VECTOR_DIM)
    result_1 = my_system.identify(probe_alice)
    print(f"👉 KẾT QUẢ CUỐI CÙNG: {result_1}")
    
    # Case 2: Người lạ (Eve) đi vào
    print("\n[TEST 2] Kẻ lạ mặt (Eve) đi vào:")
    result_2 = my_system.identify(vector_eve_unknown)
    print(f"👉 KẾT QUẢ CUỐI CÙNG: {result_2} (Chuẩn Open-set!)")

if __name__ == "__main__":
    run_simulation()