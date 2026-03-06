"""
Recognition Server (Port 3001)

Nhận hình ảnh thô từ app Android, chạy pipeline nhận diện đầy đủ:
  1. YOLOv8s-pose  → phát hiện người + keypoints + tracking (ByteTrack)
  2. ActionRecognizer (BiLSTM) → phân loại walking/non-walking
  3. YOLOv8n-seg   → trích xuất silhouette (mask) per person
  4. GaitRecognizer (ResNet18 + GEI + OpenSetMatcher) → nhận dạng dáng đi

WebSocket endpoints:
  - /camera   → Android app gửi raw JPEG frames
  - /monitor  → React UI nhận kết quả real-time
"""

import asyncio
import json
import base64
import cv2
import numpy as np
import logging
import sys
import os
from datetime import datetime
import websockets
from ultralytics import YOLO

# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

from action_classifier_module import ActionRecognizer
from gait_recognizer_module import GaitRecognizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _iou(boxA, boxB):
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------

class RecognitionPipeline:
    """
    Full ML pipeline:
      pose_model.track() → per-person bboxes + keypoints + stable track_ids
      seg_model()        → per-person silhouette masks
      ActionRecognizer   → Walking / Non-Walking
      GaitRecognizer     → identity label from gait database
    """

    def __init__(self):
        logger.info("Loading YOLOv8s-pose (detection + keypoints + ByteTrack)...")
        self.pose_model = YOLO(os.path.join(_HERE, 'yolov8s-pose.pt'))

        logger.info("Loading YOLOv8n-seg (silhouette extraction)...")
        self.seg_model = YOLO(os.path.join(_HERE, 'yolov8n-seg.pt'))

        logger.info("Loading ActionRecognizer (BiLSTM 128-hidden, 2-layer)...")
        self.action_recognizer = ActionRecognizer(
            model_path=os.path.join(_HERE, 'Behavier_recognition', 'action_classifier_128_2.pth')
        )

        logger.info("Loading GaitRecognizer (ResNet18 encoder + OpenSetMatcher)...")
        self.gait_recognizer = GaitRecognizer(
            model_path=os.path.join(_HERE, 'open_set', 'encoder_resnet.pth'),
            database_path=os.path.join(_HERE, 'database.json')
        )

        logger.info("All models loaded successfully.")

    def process_frame(self, frame_bgr):
        """
        Run full recognition pipeline on one BGR frame.

        Returns:
            vis_frame  : visualization image (numpy BGR)
            persons    : list of dicts {id, bbox, action, confidence, gait_label, is_walking}
            stats      : dict {person_count, inference_ms}
        """
        t0 = datetime.now()
        h, w = frame_bgr.shape[:2]

        # ------------------------------------------------------------------
        # Step 1 — Pose model with ByteTrack: bboxes + keypoints + track_ids
        # ------------------------------------------------------------------
        pose_res = self.pose_model.track(
            frame_bgr,
            persist=True,
            tracker='bytetrack.yaml',
            conf=0.3,
            classes=[0],  # person only
            verbose=False
        )
        pr = pose_res[0]

        tracked_persons = []  # list of (track_id:int, bbox:[x1,y1,x2,y2], kps:np(17,3))
        if pr.boxes is not None and pr.boxes.id is not None:
            bboxes    = pr.boxes.xyxy.cpu().numpy()
            track_ids = pr.boxes.id.cpu().numpy().astype(int)
            kps_data  = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else None

            for i, (box, tid) in enumerate(zip(bboxes, track_ids)):
                kps = kps_data[i] if kps_data is not None else np.zeros((17, 3), dtype=np.float32)
                tracked_persons.append((int(tid), box.tolist(), kps))

        # ------------------------------------------------------------------
        # Step 2 — Segmentation model: silhouette masks
        # ------------------------------------------------------------------
        seg_res = self.seg_model(frame_bgr, conf=0.3, classes=[0], verbose=False)
        sr = seg_res[0]

        # Combined mask canvas for visualization
        vis_mask = np.zeros((h, w), dtype=np.uint8)

        seg_pairs = []  # list of (bbox:[x1,y1,x2,y2], mask:np(h,w) uint8 0/255)
        if sr.masks is not None:
            seg_boxes = sr.boxes.xyxy.cpu().numpy()
            for i, seg_mask in enumerate(sr.masks.data):
                mask_np = (seg_mask.cpu().numpy() * 255).astype(np.uint8)
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_np = (mask_np > 127).astype(np.uint8) * 255
                vis_mask = cv2.bitwise_or(vis_mask, mask_np)
                seg_pairs.append((seg_boxes[i].tolist(), mask_np))

        # Assign best-matching silhouette mask to each tracked person (by IoU)
        person_masks = {}  # {track_id: mask_uint8}
        for (tid, bbox, _) in tracked_persons:
            best_score, best_mask = 0.0, None
            for (sbox, smask) in seg_pairs:
                score = _iou(bbox, sbox)
                if score > best_score:
                    best_score, best_mask = score, smask
            if best_score > 0.3 and best_mask is not None:
                person_masks[tid] = best_mask

        # ------------------------------------------------------------------
        # Step 3 — Per-person: action classification + gait recognition
        # ------------------------------------------------------------------
        persons_info = []
        for (tid, bbox, kps) in tracked_persons:
            pid = f"Person_{tid}"
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # --- Action ---
            self.action_recognizer.update(pid, kps.tolist(), [x1, y1, x2, y2])
            action_result = self.action_recognizer.predict(pid)
            action      = action_result['action']        # 'Walking' or 'Non-Walking'
            action_conf = action_result['confidence']
            is_walking  = (action == 'Walking')

            # --- Gait ---
            mask = person_masks.get(tid)
            gait_label = self.gait_recognizer.update(pid, frame_bgr, is_walking, mask=mask)

            persons_info.append({
                'id':         pid,
                'bbox':       [x1, y1, x2, y2],
                'action':     action,
                'confidence': round(float(action_conf), 2),
                'gait_label': gait_label or 'Unknown',
                'is_walking': is_walking,
            })

        # ------------------------------------------------------------------
        # Step 4 — Build visualization
        # ------------------------------------------------------------------
        vis_frame = self._draw_visualization(frame_bgr, vis_mask, persons_info)

        inference_ms = (datetime.now() - t0).total_seconds() * 1000
        stats = {
            'person_count': len(tracked_persons),
            'inference_ms': round(inference_ms, 1),
        }
        return vis_frame, persons_info, stats

    def _draw_visualization(self, original_bgr, mask, persons):
        """
        Composite visualization:
          - Silhouette overlay on slightly darkened original
          - Bounding boxes (green = walking, orange = not walking)
          - Text labels: person ID, action %, gait identity
        """
        silhouette_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        gray_original  = cv2.cvtColor(
            cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )
        vis = cv2.addWeighted(silhouette_bgr, 0.65, gray_original, 0.35, 0)

        for p in persons:
            x1, y1, x2, y2 = p['bbox']
            is_walking = p['is_walking']
            color = (0, 255, 0) if is_walking else (0, 165, 255)  # green / orange

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            lines = [
                f"{p['id']} | {p['action']} {p['confidence']:.0%}",
                f"Gait: {p['gait_label']}",
            ]
            for row, txt in enumerate(lines):
                ty = y1 - 5 - row * 20
                if ty < 18:
                    ty = y2 + 15 + row * 20
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), (0, 0, 0), -1)
                cv2.putText(vis, txt, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

class RecognitionServer:
    def __init__(self, host='0.0.0.0', port=3001):
        self.host = host
        self.port = port
        self.pipeline = RecognitionPipeline()
        self.monitors: set = set()   # connected React UI clients
        self.frame_count = 0

    async def handle_connection(self, websocket):
        path = websocket.request.path
        if path == '/monitor':
            await self._handle_monitor(websocket)
        else:
            # Any other path (including /camera) treated as camera client
            await self._handle_camera(websocket)

    async def _handle_monitor(self, websocket):
        """React UI client — just receives broadcast results."""
        client_addr = websocket.remote_address
        logger.info(f"Monitor connected: {client_addr}")
        self.monitors.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.monitors.discard(websocket)
            logger.info(f"Monitor disconnected: {client_addr}")

    async def _handle_camera(self, websocket):
        """Android camera client — receives raw JPEG frames."""
        client_addr = websocket.remote_address
        logger.info(f"Camera connected: {client_addr}")
        try:
            async for message in websocket:
                await self._process_message(message, client_addr)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Camera disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Camera client error: {e}", exc_info=True)

    async def _process_message(self, message, client_addr):
        """Decode frame, run ML pipeline in thread pool, broadcast to monitors."""
        try:
            data      = json.loads(message)
            frame_b64 = data.get('frame')
            if not frame_b64:
                return

            frame_bytes = base64.b64decode(frame_b64)
            np_arr      = np.frombuffer(frame_bytes, np.uint8)
            frame       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            # Run blocking inference in a thread pool so we don't stall the event loop
            loop = asyncio.get_event_loop()
            vis_frame, persons, stats = await loop.run_in_executor(
                None, self.pipeline.process_frame, frame
            )

            # Encode visualization
            _, enc = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            vis_b64 = base64.b64encode(enc.tobytes()).decode('utf-8')

            self.frame_count += 1
            payload = json.dumps({
                'type':    'frame',
                'frame':   vis_b64,
                'persons': persons,
                'metadata': {
                    'person_count': stats['person_count'],
                    'inference_ms': stats['inference_ms'],
                    'timestamp':    datetime.now().isoformat(),
                    'device':       str(client_addr[0]) if client_addr else 'unknown',
                },
            })

            # Fan-out to all monitor clients
            dead = set()
            for ws in self.monitors:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            self.monitors -= dead

            if self.frame_count % 30 == 0:
                logger.info(
                    f"Frame #{self.frame_count} | "
                    f"Persons: {stats['person_count']} | "
                    f"Inference: {stats['inference_ms']:.1f} ms"
                )

        except json.JSONDecodeError:
            logger.warning("Malformed JSON message from camera")
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)

    async def start(self):
        async with websockets.serve(self.handle_connection, self.host, self.port):
            logger.info(f"Recognition server listening on ws://{self.host}:{self.port}")
            logger.info("  Endpoints:")
            logger.info("    ws://<host>:3001/camera  — Android camera client")
            logger.info("    ws://<host>:3001/monitor — React UI monitor")
            await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3001
    server = RecognitionServer(port=port)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == '__main__':
    main()
