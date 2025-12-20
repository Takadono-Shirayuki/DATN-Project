import numpy as np
from scipy.spatial.distance import cdist

class OpenSetGaitMatcher:
    def __init__(self, metric='cosine', alpha=3.0):
        """
        Args:
            metric: 'cosine' hoặc 'euclidean'
            alpha: Hệ số nhân độ lệch chuẩn (thường là 3.0 để phủ 99.7% phân phối Gauss)
        """
        self.metric = metric
        self.alpha = alpha
        self.centroids = None      # Ma trận lưu các vector tâm của từng người
        self.labels = None         # Danh sách ID tương ứng với các tâm
        self.threshold = None      # Bán kính 'hình cầu' an toàn chung

    def fit(self, X, y):
        """
        Giai đoạn 'Huấn luyện' Open-set: Tìm Tâm và xác định Bán kính hình cầu.
        """
        unique_labels = np.unique(y)
        centroids_list = []
        all_intra_distances = []

        # 1. Tạo Tâm (Centroids) cho từng ID người dùng
        for label in unique_labels:
            class_samples = X[y == label]
            # Tâm là trung bình cộng (Mean) của tất cả các vector của cùng một người
            center = np.mean(class_samples, axis=0)
            centroids_list.append(center)
            
            # 2. Tính khoảng cách từ mỗi mẫu đến Tâm của chính nó
            # Khoảng cách này đại diện cho sự biến thiên dáng đi của người đó
            dists_to_center = cdist(class_samples, center.reshape(1, -1), metric=self.metric)
            all_intra_distances.extend(dists_to_center.flatten())

        self.centroids = np.array(centroids_list)
        self.labels = unique_labels

        # 3. Tính ngưỡng Threshold (Bán kính hình cầu Gaussian)
        # Threshold = Mean_dist + Alpha * Std_dist
        mu = np.mean(all_intra_distances)
        sigma = np.std(all_intra_distances)
        self.threshold = mu + self.alpha * sigma
        
        print(f"✅ Hệ thống đã tạo {len(unique_labels)} tâm (Centroids).")
        print(f"✨ Bán kính hình cầu an toàn (Threshold): {self.threshold:.4f}")

    def predict(self, query_X):
        """
        So khớp dựa trên khoảng cách đến Tâm.
        Returns:
            results: ID người dùng hoặc -1 nếu nằm ngoài mọi hình cầu (Unknown)
            min_dists: Khoảng cách nhỏ nhất tìm được
        """
        if self.centroids is None:
            raise ValueError("Bạn cần gọi hàm fit() trước khi predict.")

        # 1. Tính khoảng cách từ Query tới tất cả các Tâm (Centroids)
        # Thay vì so với hàng ngàn mẫu, ta chỉ so với danh sách các Tâm
        dists_to_centroids = cdist(query_X, self.centroids, metric=self.metric)
        
        min_dists = np.min(dists_to_centroids, axis=1)
        min_indices = np.argmin(dists_to_centroids, axis=1)

        results = []
        for i, d in enumerate(min_dists):
            # Nếu khoảng cách đến tâm gần nhất vẫn lớn hơn bán kính cho phép
            if d > self.threshold:
                results.append(-1) # Unknown (Nằm ngoài vùng an toàn)
            else:
                results.append(self.labels[min_indices[i]]) # Nhận diện đúng ID

        return np.array(results), min_dists

    def evaluate(self, y_true, y_pred):
        """Tính toán độ chính xác (Accuracy)"""
        return np.mean(y_true == y_pred)