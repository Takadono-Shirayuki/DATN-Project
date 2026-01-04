import json
import numpy as np
import argparse
from open_set.open_set_matcher import OpenSetGaitMatcher


def load_json_embeddings(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Expecting keys: 'embeddings' (list of lists), 'labels' (list), optional 'val_embeddings','val_labels'
    X = np.array(data.get('embeddings', []))
    y = np.array(data.get('labels', []))
    val_X = np.array(data.get('val_embeddings', [])) if 'val_embeddings' in data else None
    val_y = np.array(data.get('val_labels', [])) if 'val_labels' in data else None
    return X, y, val_X, val_y


def run_from_json(input_json, output_db='database.json', metric='cosine', prototype_method='kmeans', percentile=95, alpha=3.0):
    X, y, val_X, val_y = load_json_embeddings(input_json)
    if X.size == 0 or y.size == 0:
        raise ValueError('Input JSON must contain embeddings and labels')

    matcher = OpenSetGaitMatcher(metric=metric, alpha=alpha, filename=output_db,
                                percentile=percentile,
                                prototype_method=prototype_method)

    matcher.fit(X, y)

    # calibrate if validation provided
    if val_X is not None and val_y is not None and len(val_X) > 0:
        matcher.calibrate_thresholds(val_X, val_y, percentile=percentile)
    elif val_X is not None and len(val_X) > 0:
        matcher.calibrate_thresholds(val_X, None, percentile=percentile)

    print(f'Wrote prototypes to {output_db}')

    return output_db


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str)
    parser.add_argument('--output_db', type=str, default='database.json')
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--prototype_method', type=str, default='kmeans')
    parser.add_argument('--percentile', type=int, default=95)
    parser.add_argument('--alpha', type=float, default=3.0)
    args = parser.parse_args()

    run_from_json(args.input_json, output_db=args.output_db, metric=args.metric,
                  prototype_method=args.prototype_method, percentile=args.percentile, alpha=args.alpha)
