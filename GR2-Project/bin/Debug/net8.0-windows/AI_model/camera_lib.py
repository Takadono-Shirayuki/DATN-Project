import cv2
from common import State, skeleton, cleanup_ids, assign_id_for_person

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


def detect_pose(input_frame, output_frame, pose_model):
    active_labels = set()
    pose_results = pose_model.predict(source=input_frame, save=False, conf=0.5, verbose=False)
    for box, kp in zip(pose_results[0].boxes.xyxy, pose_results[0].keypoints.data):
        keypoints = [[round(x, 2), round(y, 2), round(c, 2)] for x, y, c in kp.tolist()]
        cropped = input_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        id, label, standing = assign_id_for_person(keypoints, cropped)
        if not standing:
            continue
        active_labels.add(id)
        if State.Camera.enable_bbox:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if State.Camera.enable_pose:
            draw_pose(output_frame, keypoints)
    cleanup_ids(active_labels)
    return output_frame
