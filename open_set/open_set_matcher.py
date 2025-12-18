import numpy as np
from scipy.spatial.distance import cdist

class OpenSetGaitMatcher:
    def __init__(self, metric='cosine', threshold=None):
        self.metric = metric
        self.threshold = threshold # Nếu là None, sẽ tự động tune khi predict

    def _calculate_auto_threshold(self, X, y, alpha=3.0):
        """Hàm nội bộ để tính ngưỡng 3-Sigma từ dữ liệu Gallery"""
        dists = cdist(X, X, metric=self.metric)
        np.fill_diagonal(dists, np.inf)
        
        pos_dists = []
        for i in range(len(X)):
            same_idx = np.where(y == y[i])[0]
            same_idx = same_idx[same_idx != i]
            if len(same_idx) > 0:
                pos_dists.append(np.min(dists[i, same_idx]))
        
        mu, sigma = np.mean(pos_dists), np.std(pos_dists)
        return mu + alpha * sigma

    def predict(self, query_X, gallery_X, gallery_y, alpha=3.0):
        """So khớp và tự động tính ngưỡng nếu cần"""
        
        # 1. Tự động tính ngưỡng (chỉ chạy nếu threshold đang là None)
        if self.threshold is None:
            self.threshold = self._calculate_auto_threshold(gallery_X, gallery_y, alpha)
            print(f"✨ Auto-tuned threshold: {self.threshold:.4f}")

        # 2. Tính khoảng cách và tìm người giống nhất
        all_dists = cdist(query_X, gallery_X, metric=self.metric)
        min_dists = np.min(all_dists, axis=1)
        min_idx = np.argmin(all_dists, axis=1)

        # 3. Phân loại Open Set
        results = [gallery_y[i] if d <= self.threshold else -1 for i, d in enumerate(min_dists)]
        
        return np.array(results), min_dists