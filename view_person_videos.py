"""
Visualize recorded person videos with pose keypoints overlay
"""
import cv2
import json
import os
import argparse
import numpy as np


# COCO keypoint pairs for skeleton
SKELETON_PAIRS = [
    (0, 1), (0, 2),           # nose to eyes
    (1, 3), (2, 4),           # eyes to ears
    (0, 5), (0, 6),           # nose to shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 6),                   # shoulders
    (5, 11), (6, 12),         # shoulders to hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]


def draw_keypoints_and_skeleton(frame, keypoints, confidence_threshold=0.3):
    """
    Draw pose keypoints and skeleton on frame
    
    Args:
        frame: BGR image
        keypoints: list of [x, y, confidence] for 17 keypoints
        confidence_threshold: minimum confidence to draw keypoint
    """
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    # Draw skeleton connections
    for pair in SKELETON_PAIRS:
        idx1, idx2 = pair
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            kp1 = keypoints[idx1]
            kp2 = keypoints[idx2]
            
            # Check confidence
            if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                x1, y1 = int(kp1[0]), int(kp1[1])
                x2, y2 = int(kp2[0]), int(kp2[1])
                
                # Draw line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints
    for kp in keypoints:
        if len(kp) >= 3 and kp[2] > confidence_threshold:
            x, y, conf = int(kp[0]), int(kp[1]), kp[2]
            
            # Color based on confidence
            color = (0, int(255 * conf), int(255 * (1 - conf)))
            cv2.circle(frame, (x, y), 3, color, -1)
    
    return frame


def visualize_video_with_keypoints(video_path, json_path, show_confidence=True):
    """
    Play video with keypoints overlay
    """
    # Load keypoints data
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keypoints_frames = data.get('frames', [])
    fps = data.get('fps', 30.0)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
    
    print(f"\nPlaying: {os.path.basename(video_path)}")
    print(f"Frames: {len(keypoints_frames)}")
    print(f"FPS: {fps}")
    print(f"Press 'q' to quit, SPACE to pause/resume")
    print("-" * 60)
    
    frame_idx = 0
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw keypoints if available
            if frame_idx < len(keypoints_frames):
                keypoints = keypoints_frames[frame_idx]
                frame = draw_keypoints_and_skeleton(frame, keypoints)
                
                # Show frame info
                cv2.putText(frame, f"Frame: {frame_idx + 1}/{len(keypoints_frames)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if show_confidence and keypoints:
                    avg_conf = np.mean([kp[2] for kp in keypoints if len(kp) >= 3])
                    cv2.putText(frame, f"Avg Confidence: {avg_conf:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            frame_idx += 1
        
        cv2.imshow('Person Video with Pose', frame)
        
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize person videos with pose keypoints')
    parser.add_argument('--video-dir', type=str, default='person_videos',
                       help='Directory containing person videos')
    parser.add_argument('--video', type=str, default=None,
                       help='Specific video file to play (optional)')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Hide confidence text')
    args = parser.parse_args()
    
    print("="*60)
    print("PERSON VIDEO VIEWER WITH POSE KEYPOINTS")
    print("="*60)
    
    if args.video:
        # Play specific video
        video_path = args.video
        json_path = os.path.splitext(video_path)[0] + '.json'
        visualize_video_with_keypoints(video_path, json_path, not args.no_confidence)
    else:
        # List all videos in directory
        if not os.path.exists(args.video_dir):
            print(f"Error: Directory not found: {args.video_dir}")
            return
        
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        
        if not video_files:
            print(f"No video files found in {args.video_dir}")
            return
        
        print(f"\nFound {len(video_files)} videos in {args.video_dir}/")
        print("\nAvailable videos:")
        for idx, video_file in enumerate(video_files, 1):
            json_file = os.path.splitext(video_file)[0] + '.json'
            json_path = os.path.join(args.video_dir, json_file)
            has_keypoints = "✓" if os.path.exists(json_path) else "✗"
            print(f"  {idx}. {video_file} (keypoints: {has_keypoints})")
        
        # Play all videos in sequence
        print(f"\nPlaying all videos in sequence...")
        for video_file in video_files:
            video_path = os.path.join(args.video_dir, video_file)
            json_path = os.path.splitext(video_path)[0] + '.json'
            visualize_video_with_keypoints(video_path, json_path, not args.no_confidence)
    
    print("\n" + "="*60)
    print("✓ Viewer closed")
    print("="*60)


if __name__ == '__main__':
    main()
