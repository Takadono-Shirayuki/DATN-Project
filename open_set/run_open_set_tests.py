import numpy as np
from time import time
import argparse
from open_set_matcher import OpenSetGaitMatcher
from data_gen import generate_gait_data


def run_test(seed=42, n_classes=10, dim=128, n_samples_per_class=20, alpha=3.0,
             percentile=95, prototype_method='kmeans', out_db='database_test.json'):
    g_X, g_y, q_k_X, q_k_y, q_u_X = generate_gait_data(n_classes=n_classes, dim=dim,
                                                       n_samples_per_class=n_samples_per_class, seed=seed)

    for metric in ['cosine', 'euclidean']:
        matcher = OpenSetGaitMatcher(metric=metric, alpha=alpha, filename=out_db, percentile=percentile)
        t0 = time()
        matcher.fit(g_X, g_y)
        fit_time = time() - t0

        # Calibrate thresholds on validation (query known)
        try:
            matcher.calibrate_thresholds(q_k_X, q_k_y, percentile=percentile)
        except Exception:
            pass

        # Known queries
        preds_k, dists_k = matcher.predict(q_k_X)
        known_acc = np.mean(preds_k == q_k_y)

        # Unknown queries
        preds_u, dists_u = matcher.predict(q_u_X)
        unknown_acc = np.mean(preds_u == -1)

        print(f"--- Metric: {metric.upper()} ---")
        print(f"Fit time: {fit_time:.4f}s")
        print(f"Known accuracy: {known_acc*100:.2f}%")
        print(f"Unknown detection accuracy: {unknown_acc*100:.2f}%")
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_samples_per_class', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--percentile', type=int, default=95)
    parser.add_argument('--prototype_method', type=str, default='kmeans')
    parser.add_argument('--out_db', type=str, default='database_test.json')
    args = parser.parse_args()

    print("Running Open-Set tests (centroid + calibrated thresholds)")
    run_test(seed=args.seed, n_classes=args.n_classes, dim=args.dim,
             n_samples_per_class=args.n_samples_per_class, alpha=args.alpha,
             percentile=args.percentile, prototype_method=args.prototype_method,
             out_db=args.out_db)
"""
Updated test runner: runs KMeans prototypes + percentile calibration
and writes prototypes to `database_test.json` by default.
"""

# (New implementation is above in file)
