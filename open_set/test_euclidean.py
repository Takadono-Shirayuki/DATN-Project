from open_set_matcher import OpenSetGaitMatcher
from data_gen import generate_gait_data

# Lấy dữ liệu
g_X, g_y, q_X, q_y, _ = generate_gait_data()

matcher = OpenSetGaitMatcher(metric='euclidean')

# Predict (Ngưỡng sẽ tự sinh ra dựa trên g_X và g_y)
preds, dists = matcher.predict(q_X, g_X, g_y)