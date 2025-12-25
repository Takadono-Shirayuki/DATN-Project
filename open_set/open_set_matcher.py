import numpy as np
import json
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class OpenSetGaitMatcher:
    def __init__(self, metric='cosine', alpha=3.0, filename="database.json"):
        self.metric = metric
        self.alpha = alpha
        self.filename = filename

    def calculate_optimal_prototypes(self, user_id, X):
        """
        Luồng Đăng ký: Tự động tìm K tối ưu và cập nhật JSON tức thì.
        X: Toàn bộ Embedding Vectors của một người dùng (ví dụ 1000 mẫu).
        """
        # --- BƯỚC 5: TÌM K TỐI ƯU (Từ 2 đến 5) ---
        best_k = 1
        if len(X) > 5: # Chỉ chia cụm nếu có đủ dữ liệu
            max_silhouette = -1
            # Thử nghiệm K để tìm độ khít cao nhất
            for k in range(2, 6):
                if len(X) <= k: break
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > max_silhouette:
                    max_silhouette = score
                    best_k = k

        # --- CHẠY K-MEANS VỚI K TỐI ƯU ---
        final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        final_labels = final_kmeans.fit_predict(X)
        centroids = final_kmeans.cluster_centers_

        prototypes = []
        for i in range(best_k):
            # Lấy các mẫu thuộc về cụm i
            cluster_samples = X[final_labels == i]
            center = centroids[i]

            # --- BƯỚC 6: TÍNH NGƯỠNG RIÊNG CHO CỤM (3-SIGMA) ---
            dists = cdist(cluster_samples, center.reshape(1, -1), metric=self.metric).flatten()
            mu = np.mean(dists)
            sigma = np.std(dists) if len(dists) > 1 else 0
            threshold = mu + self.alpha * sigma

            prototypes.append({
                "prototype_id": i,
                "centroid": center.tolist(),
                "threshold": float(threshold)
            })

        # --- BƯỚC 7: CẬP NHẬT JSON TỨC THÌ ---
        data = {}
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except: data = {}

        # Cập nhật hoặc tạo mới profile cho user_id
        data[user_id] = prototypes

        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Đã lưu {best_k} Prototypes tối ưu cho User '{user_id}' vào {self.filename}")

    def predict(self, probe_vector):
        """
        Luồng Xác thực: So khớp Probe với toàn bộ Prototypes của mọi User.
        """
        if not os.path.exists(self.filename):
            return {"user_id": "Unknown", "is_known": False}

        with open(self.filename, 'r', encoding='utf-8') as f:
            database = json.load(f)

        best_match = {"user_id": "Unknown", "distance": float('inf'), "is_known": False}

        for user_id, protos in database.items():
            for proto in protos:
                centroid = np.array(proto['centroid'])
                dist = cdist(probe_vector.reshape(1, -1), centroid.reshape(1, -1), metric=self.metric)[0][0]
                
                if dist < best_match["distance"]:
                    best_match.update({
                        "user_id": user_id,
                        "distance": dist,
                        "threshold": proto['threshold']
                    })

        if best_match["distance"] <= best_match.get("threshold", 0):
            best_match["is_known"] = True
        else:
            best_match["user_id"] = "Unknown"

        return best_match