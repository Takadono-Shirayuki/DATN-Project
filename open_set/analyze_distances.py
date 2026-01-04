import json
import numpy as np
import os
from scipy.spatial.distance import cdist

from data_gen import generate_gait_data
from open_set_matcher import OpenSetGaitMatcher


def analyze(seed=42, n_classes=10, dim=128, n_samples_per_class=20, alpha=3.0,
            metric='cosine', threshold_mode='mean_k_sigma', percentile=95, prototype_method='kmeans'):
    g_X, g_y, q_k_X, q_k_y, q_u_X = generate_gait_data(n_classes=n_classes, dim=dim,
                                                       n_samples_per_class=n_samples_per_class, seed=seed)

    matcher = OpenSetGaitMatcher(metric=metric, alpha=alpha, filename='database_test.json',
                                percentile=percentile,
                                prototype_method=prototype_method)

    # build prototypes using existing function (writes database_test.json)
    unique = np.unique(g_y)
    for lbl in unique:
        X_user = g_X[g_y == lbl]
        matcher.calculate_optimal_prototypes(int(lbl), X_user)

    # Calibrate thresholds using query_known (validation)
    try:
        matcher.calibrate_thresholds(q_k_X, q_k_y, percentile=percentile)
    except Exception as e:
        print('Calibration skipped:', e)

    # load DB
    with open('database_test.json', 'r', encoding='utf-8') as f:
        db = json.load(f)

    # collect prototypes with metadata
    proto_map = {}  # user -> list of proto dicts
    for k, v in db.items():
        proto_map[int(k)] = v

    intra_dists = []
    inter_dists = []
    thresholds = []

    # helper distance function using proto info
    def dist_vec_to_proto(vec, proto):
        cen = np.array(proto['centroid'])
        if 'cov_diag' in proto and metric == 'mahalanobis':
            arr = (vec - cen)**2
            return float(np.sqrt(np.sum(arr / (np.array(proto['cov_diag']) + 1e-6))))
        else:
            return float(cdist(vec.reshape(1, -1), cen.reshape(1, -1), metric=metric)[0][0])

    for user, protos in proto_map.items():
        samples = g_X[g_y == user]

        # assign each sample to nearest prototype of same user
        for s in samples:
            dists_same = [dist_vec_to_proto(s, p) for p in protos]
            intra_dists.append(min(dists_same))

            # distances to other users' prototypes
            other_dists = []
            for u2, protos2 in proto_map.items():
                if u2 == user: continue
                for p2 in protos2:
                    other_dists.append(dist_vec_to_proto(s, p2))
            inter_dists.append(min(other_dists))

        # collect thresholds for this user's prototypes
        for p in protos:
            thresholds.append(float(p.get('threshold', 0)))

    intra = np.array(intra_dists)
    inter = np.array(inter_dists)
    thresholds = np.array(thresholds)

    print('--- Distance stats ---')
    print(f'Intra mean: {intra.mean():.4f}, std: {intra.std():.4f}, 95p: {np.percentile(intra,95):.4f}')
    print(f'Inter mean: {inter.mean():.4f}, std: {inter.std():.4f}, 5p: {np.percentile(inter,5):.4f}')

    # False reject: intra > avg threshold
    avg_thresh = thresholds.mean() if len(thresholds)>0 else 0
    fr = np.mean(intra > avg_thresh)
    fa = np.mean(inter <= avg_thresh)
    print(f'Using average threshold {avg_thresh:.4f} -> False Reject: {fr*100:.2f}%, False Accept: {fa*100:.2f}%')

    # Suggest percentile threshold
    perc95 = np.percentile(intra, 95)
    fr_p = np.mean(intra > perc95)
    fa_p = np.mean(inter <= perc95)
    print(f'Using 95th-percentile threshold {perc95:.4f} -> FR: {fr_p*100:.2f}%, FA: {fa_p*100:.2f}%')

    # --- Evaluate matcher accuracy on probe sets ---
    # Known queries
    known_preds = []
    for vec in q_k_X:
        res = matcher.predict(vec)
        # res may return dict with 'user_id' as string
        uid = res.get('user_id') if isinstance(res, dict) else res[0]
        try:
            uid = int(uid)
        except:
            uid = -1
        known_preds.append(uid)
    known_preds = np.array(known_preds)
    known_acc = np.mean(known_preds == q_k_y)

    # Unknown queries (should be labeled as Unknown/-1)
    unknown_preds = []
    for vec in q_u_X:
        res = matcher.predict(vec)
        uid = res.get('user_id') if isinstance(res, dict) else res[0]
        try:
            uid = int(uid)
        except:
            uid = -1
        unknown_preds.append(uid)
    unknown_preds = np.array(unknown_preds)
    unknown_acc = np.mean(unknown_preds == -1)

    print('\n--- Matcher accuracy (using stored prototypes) ---')
    print(f'Known accuracy: {known_acc*100:.2f}%')
    print(f'Unknown detection accuracy: {unknown_acc*100:.2f}%')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_samples_per_class', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--threshold_mode', type=str, default='mean_k_sigma')
    parser.add_argument('--percentile', type=int, default=95)
    parser.add_argument('--prototype_method', type=str, default='kmeans')
    args = parser.parse_args()

    analyze(seed=args.seed, n_classes=args.n_classes, dim=args.dim,
            n_samples_per_class=args.n_samples_per_class, alpha=args.alpha,
            metric=args.metric, threshold_mode=args.threshold_mode,
            percentile=args.percentile, prototype_method=args.prototype_method)
