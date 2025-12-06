import cv2
import numpy as np

# Cấu trúc khung xương COCO
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_pose(frame, keypoints):
    drawed_keypoints = set()
    for i, j in skeleton:
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 > 0.3 and c2 > 0.3:
            if i not in drawed_keypoints or j not in drawed_keypoints:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            for idx, (x, y, c) in zip([i, j], [(x1, y1, c1), (x2, y2, c2)]):
                if idx not in drawed_keypoints:
                    drawed_keypoints.add(idx)
                    cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 0), -1)
                    cv2.putText(frame, f'{c:.2f}', (int(x) + 6, int(y) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if 5 in drawed_keypoints and 6 in drawed_keypoints:
        x_mid = int((keypoints[5][0] + keypoints[6][0]) // 2)
        y_mid = int((keypoints[5][1] + keypoints[6][1]) // 2)
        conf_mid = (keypoints[5][2] + keypoints[6][2]) / 2
        cv2.circle(frame, (x_mid, y_mid), 6, (255, 0, 0), -1)
        cv2.putText(frame, f'{conf_mid:.2f}', (x_mid + 6, y_mid - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.line(frame, (int(keypoints[0][0]), int(keypoints[0][1])), (x_mid, y_mid), (0, 255, 255), 2)


# detect_pose function removed - camera_module now uses detector tracking

def segmentation(input_frame, output_frame, seg_model, type = 'Foreground'):
    seg_results = seg_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
    if seg_results[0].masks is not None and seg_results[0].boxes is not None:
        masks = seg_results[0].masks.data.cpu().numpy()
        boxes = seg_results[0].boxes
        
        # Lọc chỉ masks của người (class 0 trong COCO)
        h, w = input_frame.shape[:2]
        person_masks = []
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Kiểm tra class - chỉ giữ người (class 0)
            cls = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
            if cls == 0:  # Chỉ lấy người
                person_masks.append(mask)
        
        # Kiểm tra nếu không có mask nào
        if len(person_masks) == 0:
            return output_frame
        
        # Tối ưu: Resize và combine tất cả masks cùng lúc
        # Stack masks thành tensor và resize 1 lần
        if len(person_masks) == 1:
            # Nếu chỉ có 1 người, resize trực tiếp
            combined_mask = cv2.resize(person_masks[0], (w, h))
        else:
            # Nếu nhiều người, combine trước khi resize để giảm số lần resize
            stacked_masks = np.stack(person_masks, axis=0)
            # Lấy max theo axis 0 để combine
            max_mask = np.max(stacked_masks, axis=0)
            # Resize 1 lần duy nhất
            combined_mask = cv2.resize(max_mask, (w, h))
        
        # Convert sang binary mask
        combined_mask = ((combined_mask > 0.5) * 255).astype(np.uint8)
        
        if type == 'Foreground':
            inverse_mask = cv2.bitwise_not(combined_mask)
            output_frame[inverse_mask == 255] = (0, 0, 0)
        elif type == 'Binary':
            inverse_mask = cv2.bitwise_not(combined_mask)
            output_frame[:] = (255, 255, 255)
            output_frame[inverse_mask == 255] = (0, 0, 0)
    return output_frame
