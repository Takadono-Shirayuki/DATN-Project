
import os
import cv2
import glob

# Quét toàn bộ video trong raw dataset (bao gồm thư mục con)
RAW_DATASET_DIR = 'raw dataset'
MIN_DURATION = 2.0  # giây

def is_video_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg']

deleted = 0
checked = 0

print("\n" + "="*60)
print(f"Cleaning up short videos (< {MIN_DURATION}s) in '{RAW_DATASET_DIR}' ...")
print("="*60)

for filepath in glob.glob(f'{RAW_DATASET_DIR}/**/*', recursive=True):
    if not os.path.isfile(filepath) or not is_video_file(filepath):
        continue
    try:
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        checked += 1
        if duration < MIN_DURATION:
            os.remove(filepath)
            print(f'❌ Deleted: {filepath} ({duration:.2f}s)')
            deleted += 1
    except Exception as e:
        print(f'Error processing {filepath}: {e}')

print(f'\n✅ Done! Checked {checked} videos, deleted {deleted} videos < {MIN_DURATION}s.')
