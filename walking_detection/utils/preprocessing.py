import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    from detector import YOLOPersonDetector
except ImportError:
    print("Error: Could not find detector.py in parent directory.")
    sys.exit(1)

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = str(SCRIPT_DIR.parent / "dataset")
OUTPUT_DIR = str(SCRIPT_DIR.parent / "dataset_processed")
IMG_SIZE = 224
SEQUENCE_LENGTH = 32
MODEL_PATH = "yolov8n.pt"

def is_video_processed(video_path, output_folder):
    """Check if video has already been processed by checking output folder existence"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_folder, video_name)
    
    if not os.path.exists(save_path):
        return False
    
    # Check if folder has expected number of frames
    frame_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    
    # Consider processed if folder exists and has at least some frames
    # (Some videos might have fewer frames than SEQUENCE_LENGTH)
    return len(frame_files) > 0


def process_video_optimized(video_path, output_folder, detector, skip_if_exists=True):
    # Check if already processed
    if skip_if_exists and is_video_processed(video_path, output_folder):
        return "skipped"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "failed"

    # A. Get total frames to calculate indices
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return "failed"

    # B. Calculate exactly which indices we need (Smart Sampling)
    # We only want 16 frames evenly distributed
    if total_frames < SEQUENCE_LENGTH:
        indices_to_process = list(range(total_frames))
    else:
        indices_to_process = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
    
    # Convert to set for O(1) lookup
    indices_set = set(indices_to_process)
    
    frames_buffer = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # C. OPTIMIZATION: Only run YOLO on the specific frames we need
        if frame_idx in indices_set:
            detections = detector.detect(frame)
            
            person_img = None
            
            if len(detections) > 0:
                # Get biggest box
                best_det = max(detections, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
                x1, y1, x2, y2 = int(best_det[0]), int(best_det[1]), int(best_det[2]), int(best_det[3])
                
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size != 0:
                    person_img = cv2.resize(person_crop, (IMG_SIZE, IMG_SIZE))
            
            if person_img is not None:
                frames_buffer.append(person_img)
        
        frame_idx += 1
        # Early exit if we have passed the last needed frame
        if len(indices_to_process) > 0 and frame_idx > indices_to_process[-1]:
            break
            
    cap.release()

    # D. Final check and padding
    if len(frames_buffer) == 0: 
        return "failed"
    
    final_frames = []
    if len(frames_buffer) < SEQUENCE_LENGTH:
        final_frames = frames_buffer + [frames_buffer[-1]] * (SEQUENCE_LENGTH - len(frames_buffer))
    else:
        final_frames = frames_buffer[:SEQUENCE_LENGTH]

    # Save to disk
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_folder, video_name)
    os.makedirs(save_path, exist_ok=True)
    
    for i, frame in enumerate(final_frames):
        cv2.imwrite(f"{save_path}/frame_{i:02d}.jpg", frame)
    
    return "success"

def main():
    print("--- STARTING OPTIMIZED PREPROCESSING ---")
    
    # Check model path
    model_to_use = MODEL_PATH if os.path.exists(MODEL_PATH) else 'yolov8n.pt'
    
    # IMPORTANT: use_tracking=False
    # Because we are skipping frames, tracking won't work well and isn't needed.
    # This also fixes the 'lap' error.
    detector = YOLOPersonDetector(model_path=model_to_use, conf_thresh=0.4, use_tracking=False)
    
    for split in ['train', 'val', 'test']:
        for label in ['walking', 'non_walking']:
            input_path = os.path.join(INPUT_DIR, split, label)
            output_path = os.path.join(OUTPUT_DIR, split, label)
            
            if not os.path.exists(input_path): continue
                
            print(f"Processing {split}/{label}...")
            video_files = [f for f in os.listdir(input_path) if f.endswith('.mp4')]
            
            # Count stats
            skipped_count = 0
            processed_count = 0
            failed_count = 0
            
            for vid in tqdm(video_files, desc=f"{split}/{label}"):
                result = process_video_optimized(
                    os.path.join(input_path, vid), 
                    output_path, 
                    detector,
                    skip_if_exists=True
                )
                
                if result == "skipped":
                    skipped_count += 1
                elif result == "success":
                    processed_count += 1
                else:
                    failed_count += 1
            
            print(f"  Completed: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")

if __name__ == "__main__":
    main()