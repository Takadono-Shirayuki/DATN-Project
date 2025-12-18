import numpy as np

def generate_gait_data(n_classes=10, dim=128, n_samples_per_class=20, seed=None):
    """
    Sinh dữ liệu giả lập 'Realistic' cho Gait Recognition.
    
    Cải tiến so với bản cũ:
    1. Anisotropic Clusters: Dữ liệu hình Elip dẹt, không phải hình cầu tròn.
    2. Hard Negatives: Tạo ra các cặp người có dáng đi rất giống nhau.
    3. Viewpoint/Covariate Shift: Giả lập sự thay đổi do góc quay camera.
    """
    np.random.seed(seed)
    
    # --- CẤU HÌNH REALISTIC ---
    n_imposters = 5              # Số lượng người lạ
    hard_negative_ratio = 0.3    # 30% số class sẽ rất giống nhau (Hard Pairs)
    
    # 1. Tạo tâm (Centers) cơ bản trên mặt cầu
    total_classes = n_classes + n_imposters
    centers = np.random.randn(total_classes, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # [REALISM 1] Tạo Hard Negatives (Người giống người)
    # Thay vì random hoàn toàn, ta ép một số class nằm rất gần nhau
    n_hard = int(total_classes * hard_negative_ratio)
    for i in range(0, n_hard, 2):
        if i + 1 < total_classes:
            # Class i+1 sẽ là bản sao của Class i + một chút nhiễu nhỏ
            # Mô phỏng 2 người có dáng đi giống hệt nhau
            drift = np.random.normal(0, 0.05, dim) # Nhiễu rất nhỏ
            centers[i+1] = centers[i] + drift
            # Chuẩn hóa lại để nằm trên mặt cầu
            centers[i+1] = centers[i+1] / np.linalg.norm(centers[i+1])

    gallery_X, gallery_y = [], []
    query_known_X, query_known_y = [], []
    query_unknown_X = []

    # [REALISM 2] Tạo ma trận hiệp phương sai ngẫu nhiên cho từng class (Shape of cluster)
    # Mỗi người sẽ có độ biến thiên khác nhau ở các chiều khác nhau -> Tạo hình Elip
    class_covariances = []
    for _ in range(total_classes):
        # Tạo scale ngẫu nhiên cho từng chiều: có chiều co lại (0.02), có chiều dãn ra (0.15)
        scale_vector = np.random.uniform(0.02, 0.15, dim) 
        class_covariances.append(scale_vector)

    # [REALISM 3] Giả lập Covariate Shift (Ví dụ: Góc quay Camera)
    # Giả sử có 2 góc quay: 0 độ và 90 độ. 
    # Góc quay làm vector bị lệch đi một đoạn cố định đối với TẤT CẢ mọi người.
    view_shift_vector = np.random.normal(0, 0.05, dim)

    # --- HÀM SINH MẪU CHO 1 CLASS ---
    def generate_samples_for_class(cls_idx, n_samples):
        center = centers[cls_idx]
        cov_scale = class_covariances[cls_idx]
        
        # Sinh nhiễu cơ bản (Standard Normal)
        noise = np.random.randn(n_samples, dim)
        
        # Biến hình cầu thành hình Elip (nhân với covariance scale)
        weighted_noise = noise * cov_scale
        
        # Tạo mẫu
        samples = center + weighted_noise
        
        # Thêm Viewpoint Shift cho 50% số mẫu (giả sử nửa này quay góc khác)
        # Điều này làm dữ liệu của 1 người bị tách thành 2 cụm nhỏ gần nhau
        shift_mask = np.random.rand(n_samples, 1) > 0.5
        samples += shift_mask * view_shift_vector
        
        # Chuẩn hóa về mặt cầu (đặc trưng của bài toán Cosine/ArcFace)
        # Tuy nhiên, ta giữ lại một chút biến động độ dài (magnitude) để Euclide vẫn hoạt động
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / norms 
        
        # [Magnitude Variation] Thêm biến động độ dài (mô phỏng ảnh mờ/rõ)
        # Ảnh mờ thường có norm nhỏ hơn (nếu chưa L2 normalize) hoặc phân bố kém tự tin hơn
        magnitudes = np.random.uniform(0.9, 1.1, (n_samples, 1))
        return samples * magnitudes

    # 2. Sinh dữ liệu Known
    for cls_id in range(n_classes):
        samples = generate_samples_for_class(cls_id, n_samples_per_class)
        
        split = n_samples_per_class // 2
        gallery_X.append(samples[:split])
        gallery_y.extend([cls_id] * split)
        
        query_known_X.append(samples[split:])
        query_known_y.extend([cls_id] * (n_samples_per_class - split))

    # 3. Sinh dữ liệu Unknown
    for cls_id in range(n_classes, total_classes):
        # Unknown chỉ cần sinh ít mẫu để test
        samples = generate_samples_for_class(cls_id, n_samples_per_class // 2)
        query_unknown_X.append(samples)

    # Convert to numpy
    gallery_X = np.vstack(gallery_X)
    gallery_y = np.array(gallery_y)
    query_known_X = np.vstack(query_known_X)
    query_known_y = np.array(query_known_y)
    query_unknown_X = np.vstack(query_unknown_X)

    print(f"[DataGen - Realistic Mode] Generated Data (Seed={seed})")
    print(f" - Features: Includes Hard Negatives & Viewpoint Shifts")
    print(f" - Gallery: {gallery_X.shape}")
    print(f" - Query Known: {query_known_X.shape}")
    print(f" - Query Unknown: {query_unknown_X.shape}")
    
    return gallery_X, gallery_y, query_known_X, query_known_y, query_unknown_X

if __name__ == "__main__":
    generate_gait_data()