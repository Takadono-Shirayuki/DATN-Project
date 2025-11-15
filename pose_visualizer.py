"""
Pose Visualizer - Mix keypoints lên frame để visualize skeleton
"""
import cv2
import numpy as np
import json


# COCO 17 keypoints skeleton connections
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Đầu: nose-eyes, eyes-ears
    [5, 6],  # Vai: left-right shoulder
    [5, 7], [7, 9],  # Tay trái: shoulder-elbow-wrist
    [6, 8], [8, 10],  # Tay phải: shoulder-elbow-wrist
    [5, 11], [6, 12],  # Thân: shoulders-hips
    [11, 12],  # Hông: left-right hip
    [11, 13], [13, 15],  # Chân trái: hip-knee-ankle
    [12, 14], [14, 16]  # Chân phải: hip-knee-ankle
]

# Màu cho skeleton (BGR)
COLOR_SKELETON = (0, 255, 0)  # Xanh lá
COLOR_KEYPOINT = (0, 0, 255)  # Đỏ


def draw_keypoints_on_frame(frame, keypoints, confidence_threshold=0.5, 
                             skeleton=None, color_skeleton=None, color_keypoint=None):
    """
    Vẽ keypoints và skeleton lên 1 frame
    
    Args:
        frame: numpy array (H, W, 3) BGR
        keypoints: list of [x, y, confidence] với 17 keypoints COCO
        confidence_threshold: ngưỡng confidence để vẽ keypoint
        skeleton: list các cặp [idx1, idx2] để vẽ xương (mặc định dùng SKELETON)
        color_skeleton: màu của xương BGR (mặc định xanh lá)
        color_keypoint: màu của keypoint BGR (mặc định đỏ)
    
    Returns:
        frame với keypoints và skeleton đã vẽ
    """
    if frame is None or keypoints is None or len(keypoints) == 0:
        return frame
    
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Sử dụng default values
    if skeleton is None:
        skeleton = SKELETON
    if color_skeleton is None:
        color_skeleton = COLOR_SKELETON
    if color_keypoint is None:
        color_keypoint = COLOR_KEYPOINT
    
    # Convert keypoints to numpy array
    kp_array = np.array(keypoints)  # shape: (17, 3)
    
    # Vẽ skeleton lines
    for idx1, idx2 in skeleton:
        if idx1 < len(kp_array) and idx2 < len(kp_array):
            x1, y1, c1 = kp_array[idx1]
            x2, y2, c2 = kp_array[idx2]
            
            # Chỉ vẽ nếu cả 2 keypoints có confidence đủ cao
            if c1 > confidence_threshold and c2 > confidence_threshold:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                
                # Kiểm tra tọa độ hợp lệ
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(frame, pt1, pt2, color_skeleton, 2)
    
    # Vẽ keypoints
    for x, y, c in kp_array:
        if c > confidence_threshold:
            pt = (int(x), int(y))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, pt, 3, color_keypoint, -1)
    
    return frame


def mix_video_with_keypoints(video_path, json_path, output_path, 
                              confidence_threshold=0.5, codec='mp4v',
                              skeleton=None, color_skeleton=None, color_keypoint=None):
    """
    Mix toàn bộ video với keypoints từ JSON file
    
    Args:
        video_path: đường dẫn đến file video input
        json_path: đường dẫn đến file JSON chứa keypoints
        output_path: đường dẫn đến file video output
        confidence_threshold: ngưỡng confidence để vẽ keypoint
        codec: codec để encode video (mặc định 'mp4v')
        skeleton: list các cặp [idx1, idx2] để vẽ xương (mặc định dùng SKELETON)
        color_skeleton: màu của xương BGR (mặc định xanh lá)
        color_keypoint: màu của keypoint BGR (mặc định đỏ)
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    # Đọc JSON keypoints
    try:
        with open(json_path, 'r') as f:
            kp_data = json.load(f)
        frames_kp = kp_data.get('frames', [])
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return False
    
    # Mở video input
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error creating video writer: {output_path}")
        cap.release()
        return False
    
    print(f"Processing video: {total_frames} frames, {fps} fps, {width}x{height}")
    
    # Process từng frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lấy keypoints tại frame này (nếu có)
        if frame_idx < len(frames_kp):
            keypoints = frames_kp[frame_idx]
            if keypoints is not None and len(keypoints) > 0:
                frame = draw_keypoints_on_frame(
                    frame, keypoints, confidence_threshold,
                    skeleton, color_skeleton, color_keypoint
                )
        
        # Ghi frame
        out.write(frame)
        frame_idx += 1
        
        # Progress
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Video saved to: {output_path}")
    return True


# Test function
if __name__ == "__main__":
    # Example usage
    
    # Test với 1 frame
    # frame = cv2.imread('test.jpg')
    # keypoints = [[100, 100, 0.9], [110, 105, 0.8], ...]  # 17 keypoints
    # result = draw_keypoints_on_frame(frame, keypoints)
    # cv2.imwrite('result.jpg', result)
    
    # Test với video
    # success = mix_video_with_keypoints(
    #     'input.mp4',
    #     'input.json',
    #     'output_with_skeleton.mp4'
    # )
    
    print("Pose visualizer module ready to use")
