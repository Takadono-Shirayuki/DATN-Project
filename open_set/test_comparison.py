import numpy as np
import pandas as pd
from open_set_matcher import OpenSetGaitMatcher
from data_gen import generate_gait_data
from tqdm import tqdm # Thư viện để hiện thanh tiến trình (nếu có)

def run_large_scale_test(n_trials=100):
    print(f"🚀 Bắt đầu thử nghiệm quy mô lớn: {n_trials} bộ test...")
    
    # Danh sách lưu kết quả của từng lần chạy
    cosine_known_scores = []
    cosine_unknown_scores = []
    euclidean_known_scores = []
    euclidean_unknown_scores = []

    for i in tqdm(range(n_trials), desc="Đang chạy mô phỏng"):
        # 1. Sinh dữ liệu mới hoàn toàn (Realistic Mode) cho mỗi vòng lặp
        # Không truyền seed để mỗi lần là một bộ dữ liệu khác nhau
        g_X, g_y, q_k_X, q_k_y, q_u_X = generate_gait_data(n_classes=10, n_samples_per_class=20)

        # --- THỬ NGHIỆM VỚI COSINE ---
        matcher_cos = OpenSetGaitMatcher(metric='cosine')
        # Lần chạy này sẽ tự động tune threshold dựa trên g_X, g_y
        p_k_cos, _ = matcher_cos.predict(q_k_X, g_X, g_y, alpha=3.0)
        p_u_cos, _ = matcher_cos.predict(q_u_X, g_X, g_y, alpha=3.0)
        
        cosine_known_scores.append(np.mean(p_k_cos == q_k_y))
        cosine_unknown_scores.append(np.mean(p_u_cos == -1))

        # --- THỬ NGHIỆM VỚI EUCLIDEAN ---
        matcher_euc = OpenSetGaitMatcher(metric='euclidean')
        p_k_euc, _ = matcher_euc.predict(q_k_X, g_X, g_y, alpha=3.0)
        p_u_euc, _ = matcher_euc.predict(q_u_X, g_X, g_y, alpha=3.0)
        
        euclidean_known_scores.append(np.mean(p_k_euc == q_k_y))
        euclidean_unknown_scores.append(np.mean(p_u_euc == -1))

    # 2. Tổng hợp kết quả trung bình
    summary = {
        "Metric": ["COSINE", "EUCLIDEAN"],
        "Known Acc (Avg)": [
            f"{np.mean(cosine_known_scores)*100:.2f}%",
            f"{np.mean(euclidean_known_scores)*100:.2f}%"
        ],
        "Unknown Acc (Avg)": [
            f"{np.mean(cosine_unknown_scores)*100:.2f}%",
            f"{np.mean(euclidean_unknown_scores)*100:.2f}%"
        ],
        "Stability (Std Dev)": [
            f"{np.std(cosine_known_scores):.4f}",
            f"{np.std(euclidean_known_scores):.4f}"
        ]
    }

    # 3. Hiển thị bảng so sánh
    df = pd.DataFrame(summary)
    print("\n" + "="*60)
    print(f"BẢNG SO SÁNH TỔNG THỂ SAU {n_trials} LẦN CHẠY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Kết luận dựa trên số liệu
    cos_overall = (np.mean(cosine_known_scores) + np.mean(cosine_unknown_scores)) / 2
    euc_overall = (np.mean(euclidean_known_scores) + np.mean(euclidean_unknown_scores)) / 2
    
    winner = "COSINE" if cos_overall > euc_overall else "EUCLIDEAN"
    print(f"Better: {winner}")

if __name__ == "__main__":
    run_large_scale_test(n_trials=100)