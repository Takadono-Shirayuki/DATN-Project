import numpy as np
import json
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class OpenSetGaitMatcher:
    def __init__(self, metric='cosine', alpha=3.0, filename="database.json", percentile=95):
        """
        Simplified matcher: use KMeans prototypes and percentile-based thresholds.
        percentile: percentile used for per-prototype threshold calibration (default 95).
        """
        self.metric = metric
        self.alpha = alpha
        self.filename = filename
        self.percentile = int(percentile)

    def calculate_optimal_prototypes(self, user_id, X):
        """
        Luồng Đăng ký: Tự động tìm K tối ưu và cập nhật JSON tức thì.
        X: Toàn bộ Embedding Vectors của một người dùng (ví dụ 1000 mẫu).
        """
        # KMeans path only (simplified)
        prototypes = []
        best_k = 1
        if len(X) > 5:
            max_silhouette = -1
            for k in range(2, 6):
                if len(X) <= k:
                    break
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > max_silhouette:
                    max_silhouette = score
                    best_k = k

        final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        final_labels = final_kmeans.fit_predict(X)
        centroids = final_kmeans.cluster_centers_

        for i in range(best_k):
            cluster_samples = X[final_labels == i]
            center = centroids[i]
            dists = cdist(cluster_samples, center.reshape(1, -1), metric=self.metric).flatten()

            # Use percentile-based threshold (default: self.percentile)
            threshold = float(np.percentile(dists, self.percentile)) if len(dists) > 0 else 0.0

            prototypes.append({
                "prototype_id": int(i),
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
        
        print(f"✅ Đã lưu {len(prototypes)} Prototypes tối ưu cho User '{user_id}' vào {self.filename}")

    def fit(self, X, y):
        """
        Compute prototypes for each label from arrays X,y and save to DB.
        y can be integers or strings (label names).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        unique_labels = np.unique(y)
        for lbl in unique_labels:
            X_user = X[y == lbl]
            # Giữ nguyên label (string hoặc int) thay vì convert sang int
            user_id = str(lbl)
            self.calculate_optimal_prototypes(user_id, X_user)

    def _mahalanobis_dist(self, x, centroid, cov_diag):
        arr = (x - centroid)**2
        return float(np.sqrt(np.sum(arr / (np.array(cov_diag) + 1e-6))))

    def predict(self, probe_vector_or_X):
        """
        Supports single vector or batch. Returns dict for single, or (preds,dists) for batch.
        """
        if not os.path.exists(self.filename):
            if isinstance(probe_vector_or_X, np.ndarray) and probe_vector_or_X.ndim == 2:
                n = probe_vector_or_X.shape[0]
                return np.array(["Unknown"] * n), np.array([float('inf')] * n)
            return {"user_id": "Unknown", "is_known": False}

        with open(self.filename, 'r', encoding='utf-8') as f:
            database = json.load(f)

        def predict_single(vec):
            best_match = {"user_id": "Unknown", "distance": float('inf'), "is_known": False, "threshold": 0}
            for user_id, protos in database.items():
                for proto in protos:
                    centroid = np.array(proto['centroid'])
                    if 'cov_diag' in proto and self.metric == 'mahalanobis':
                        dist = self._mahalanobis_dist(vec, centroid, proto['cov_diag'])
                    else:
                        use_metric = self.metric
                        if self.metric == 'mahalanobis':
                            use_metric = 'euclidean'
                        dist = cdist(vec.reshape(1, -1), centroid.reshape(1, -1), metric=use_metric)[0][0]

                    if dist < best_match['distance']:
                        best_match.update({
                            'user_id': user_id,
                            'distance': float(dist),
                            'threshold': float(proto.get('threshold', 0))
                        })

            if best_match['distance'] <= best_match.get('threshold', 0):
                best_match['is_known'] = True
            else:
                best_match['user_id'] = "Unknown"
            return best_match

        arr = np.asarray(probe_vector_or_X)
        if arr.ndim == 1:
            return predict_single(arr)

        preds = []
        dists = []
        for vec in arr:
            m = predict_single(vec)
            preds.append(m['user_id'])
            dists.append(m['distance'])

        return np.array(preds), np.array(dists)

    def fit(self, X, y):
        """
        Fit the matcher from embeddings X and labels y.
        X: np.ndarray shape (n_samples, dim)
        y: array-like labels (n_samples,)
        This will compute prototypes per label and save to self.filename.
        y can be integers or strings (label names).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        unique_labels = np.unique(y)
        for lbl in unique_labels:
            X_user = X[y == lbl]
            # Giữ nguyên label (string hoặc int) thay vì convert sang int
            user_id = str(lbl)
            self.calculate_optimal_prototypes(user_id, X_user)

    def calibrate_thresholds(self, val_X, val_y=None, percentile=None):
        """
        Calibrate per-prototype thresholds using validation embeddings.
        If val_y provided, only use samples with matching label to compute intra-class dists.
        percentile: if provided, use this percentile of intra distances per-prototype; otherwise use self.percentile.
        This updates thresholds in self.filename (JSON).
        """
        if percentile is None:
            percentile = getattr(self, 'percentile', 95)

        if not os.path.exists(self.filename):
            raise FileNotFoundError(self.filename)

        with open(self.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        val_X = np.asarray(val_X)
        if val_y is not None:
            val_y = np.asarray(val_y)

        # For each prototype, collect distances from relevant val samples
        for user_str, protos in list(data.items()):
            user = None
            try:
                user = int(user_str)
            except:
                user = user_str

            for i, proto in enumerate(protos):
                centroid = np.array(proto['centroid'])
                # select validation samples for this user if labels available
                if val_y is not None:
                    mask = (val_y == user)
                    samples = val_X[mask]
                else:
                    samples = val_X

                if samples is None or len(samples) == 0:
                    # no samples to calibrate this proto; skip
                    continue

                # compute distances to this centroid
                if 'cov_diag' in proto and self.metric == 'mahalanobis':
                    arr = (samples - centroid)**2
                    dists = np.sqrt(np.sum(arr / (np.array(proto['cov_diag']) + 1e-6), axis=1))
                else:
                    dists = cdist(samples, centroid.reshape(1, -1), metric=('euclidean' if self.metric=='mahalanobis' else self.metric)).flatten()

                # set threshold as percentile
                if len(dists) > 0:
                    new_thr = float(np.percentile(dists, percentile))
                    proto['threshold'] = new_thr

        # save back
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Calibrated thresholds using percentile={percentile} and saved to {self.filename}")