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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import websockets
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Unicode / Vietnamese font helper
# ---------------------------------------------------------------------------

_FONT_CACHE: dict = {}

def _load_unicode_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Load a TrueType font that supports Vietnamese characters.

    Tries common system font paths (Windows then Linux).  Falls back to
    PIL's built-in bitmap font if no TrueType font is found.
    """
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\times.ttf",
        # Linux / macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    font = None
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font

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


def _crop_silhouette_square(mask_full, bbox):
    """
    Crop a square silhouette patch from the full-frame mask.

    Center  = center of the bounding box.
    Side    = max(bbox_width, bbox_height)  → square crop.
    Regions that fall outside the frame are filled with 0 (black).

    Args:
        mask_full : np.ndarray (H, W) uint8 — full-frame binary mask (0 or 255)
        bbox      : [x1, y1, x2, y2] ints

    Returns:
        np.ndarray (side, side) uint8
    """
    H, W = mask_full.shape[:2]
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2

    # Crop boundaries in the full frame
    src_x0 = cx - half
    src_y0 = cy - half
    src_x1 = cx + half
    src_y1 = cy + half

    side = src_x1 - src_x0  # == 2 * half
    canvas = np.zeros((side, side), dtype=np.uint8)

    # Intersection of crop window with actual frame
    fx0 = max(src_x0, 0)
    fy0 = max(src_y0, 0)
    fx1 = min(src_x1, W)
    fy1 = min(src_y1, H)

    if fx1 > fx0 and fy1 > fy0:
        # Corresponding destination in canvas
        dx0 = fx0 - src_x0
        dy0 = fy0 - src_y0
        dx1 = dx0 + (fx1 - fx0)
        dy1 = dy0 + (fy1 - fy0)
        canvas[dy0:dy1, dx0:dx1] = mask_full[fy0:fy1, fx0:fx1]

    return canvas


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
        self._use_gpu = torch.cuda.is_available()
        device_str = 'cuda:0' if self._use_gpu else 'cpu'

        logger.info(f"Device: {'GPU (CUDA)' if self._use_gpu else 'CPU'}")

        logger.info("Loading YOLOv8s-pose (detection + keypoints + ByteTrack)...")
        self.pose_model = YOLO(os.path.join(_HERE, 'yolov8s-pose.pt'))
        self.pose_model.to(device_str)

        logger.info("Loading YOLOv8n-seg (silhouette extraction)...")
        self.seg_model = YOLO(os.path.join(_HERE, 'yolov8n-seg.pt'))
        self.seg_model.to(device_str)

        logger.info("Loading ActionRecognizer (BiLSTM 128-hidden, 2-layer)...")
        self.action_recognizer = ActionRecognizer(
            model_path=os.path.join(_HERE, 'Behavier_recognition', 'action_classifier_128_2.pth')
        )

        logger.info("Loading GaitRecognizer (ResNet18 encoder + OpenSetMatcher)...")
        self.gait_recognizer = GaitRecognizer(
            model_path=os.path.join(_HERE, 'open_set', 'encoder_resnet.pth'),
            database_path=os.path.join(_HERE, 'database.json')
        )

        # Warmup both models so first-frame latency is low
        if self._use_gpu:
            _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.pose_model(_dummy, imgsz=640, verbose=False, half=True)
            self.seg_model(_dummy, imgsz=640, verbose=False, half=True)
            logger.info("GPU warmup done.")

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

        # Resize frame to max 640px wide before inference to reduce GPU transfer cost.
        # Scale factor is used to map detections back to original coordinates.
        infer_w = 640
        if w > infer_w:
            scale = infer_w / w
            infer_h = int(h * scale)
            infer_frame = cv2.resize(frame_bgr, (infer_w, infer_h))
        else:
            scale = 1.0
            infer_frame = frame_bgr
        ih, iw = infer_frame.shape[:2]

        half_flag = self._use_gpu

        # ------------------------------------------------------------------
        # Step 1 — Pose model with ByteTrack: bboxes + keypoints + track_ids
        # ------------------------------------------------------------------
        pose_res = self.pose_model.track(
            infer_frame,
            persist=True,
            tracker='bytetrack.yaml',
            conf=0.3,
            classes=[0],  # person only
            verbose=False,
            half=half_flag,
            imgsz=640,
        )
        pr = pose_res[0]

        tracked_persons = []  # list of (track_id:int, bbox:[x1,y1,x2,y2], kps:np(17,3))
        has_walking_candidate = False
        if pr.boxes is not None and pr.boxes.id is not None:
            bboxes    = pr.boxes.xyxy.cpu().numpy()
            track_ids = pr.boxes.id.cpu().numpy().astype(int)
            kps_data  = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else None

            for i, (box, tid) in enumerate(zip(bboxes, track_ids)):
                kps = kps_data[i] if kps_data is not None else np.zeros((17, 3), dtype=np.float32)
                # Scale bbox back to original frame coordinates
                sx1, sy1, sx2, sy2 = box
                orig_box = [sx1 / scale, sy1 / scale, sx2 / scale, sy2 / scale]
                tracked_persons.append((int(tid), orig_box, kps))

            # Quick pre-check: are any persons likely walking (rough heuristic: bbox tall)
            has_walking_candidate = len(tracked_persons) > 0

        # ------------------------------------------------------------------
        # Step 2 — Segmentation model: only run when there are tracked persons
        # ------------------------------------------------------------------
        seg_pairs = []  # list of (bbox:[x1,y1,x2,y2], mask:np(h,w) uint8 0/255)
        if has_walking_candidate:
            seg_res = self.seg_model(infer_frame, conf=0.3, classes=[0], verbose=False,
                                     half=half_flag, imgsz=640)
            sr = seg_res[0]

            if sr.masks is not None:
                seg_boxes = sr.boxes.xyxy.cpu().numpy()
                for i, seg_mask in enumerate(sr.masks.data):
                    mask_np = (seg_mask.cpu().numpy() * 255).astype(np.uint8)
                    # Resize mask to original frame size
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_np = (mask_np > 127).astype(np.uint8) * 255
                    # Scale seg bbox back to original coords
                    sb = seg_boxes[i] / scale
                    seg_pairs.append((sb.tolist(), mask_np))

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
            if mask is not None:
                mask = _crop_silhouette_square(mask, [x1, y1, x2, y2])
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
        vis_frame = self._draw_visualization(frame_bgr, persons_info)

        inference_ms = (datetime.now() - t0).total_seconds() * 1000
        stats = {
            'person_count': len(tracked_persons),
            'inference_ms': round(inference_ms, 1),
        }
        return vis_frame, persons_info, stats

    def _draw_visualization(self, original_bgr, persons):
        """
        Draw bounding boxes and recognition labels on the original frame.
          - Bounding boxes (green = walking, orange = not walking)
          - Text labels: person ID, action %, gait identity
        Uses PIL for text rendering so Vietnamese (Unicode) characters display correctly.
        """
        vis = original_bgr.copy()

        # --- Bounding boxes (OpenCV, no Unicode needed) ---
        for p in persons:
            x1, y1, x2, y2 = p['bbox']
            color = (0, 255, 0) if p['is_walking'] else (0, 165, 255)  # green / orange
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # --- Text labels (PIL for full Unicode / Vietnamese support) ---
        pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = _load_unicode_font(16)

        for p in persons:
            x1, y1, x2, y2 = p['bbox']
            color_bgr = (0, 255, 0) if p['is_walking'] else (0, 165, 255)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR → RGB
            if p['gait_label'] == 'Unknown':
                lines = [
                    f"{p['action']} {p['confidence']:.0%}",
                    f"Gait: Unknown {p['id']}",
                ]
            else:
                lines = [
                    f"Gait: {p['gait_label']}"
                ]
            for row, txt in enumerate(lines):
                # ty acts as the baseline (bottom of text), matching original layout
                ty = y1 - 5 - row * 20
                if ty < 18:
                    ty = y2 + 15 + row * 20

                # Measure text with PIL
                bbox_px = draw.textbbox((0, 0), txt, font=font)
                tw = bbox_px[2] - bbox_px[0]
                th = bbox_px[3] - bbox_px[1]

                # Black background rectangle
                draw.rectangle(
                    [x1, ty - th - 3, x1 + tw + 4, ty + 3],
                    fill=(0, 0, 0)
                )
                # Text: PIL uses top-left origin, so top = ty - th
                draw.text((x1 + 2, ty - th), txt, font=font, fill=color_rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Per-camera session
# ---------------------------------------------------------------------------

class _CameraSession:
    """Isolated processing session for one camera or video source.

    Each session owns its own RecognitionPipeline so ByteTrack IDs and
    per-person action/gait buffers are never shared across sources.
    The pipeline is loaded lazily in a background thread when the session
    starts, so the WebSocket receive loop is never blocked.
    """

    def __init__(self, device_label: str, server: 'RecognitionServer'):
        self.device_label = device_label
        self._server = server
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run(), name=f'session-{self.device_label}')

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _run(self):
        loop = asyncio.get_event_loop()
        logger.info(f"[{self.device_label}] Loading pipeline…")
        try:
            pipeline = await loop.run_in_executor(
                self._server._executor, RecognitionPipeline
            )
        except Exception as e:
            logger.error(f"[{self.device_label}] Pipeline load failed: {e}", exc_info=True)
            return
        logger.info(f"[{self.device_label}] Pipeline ready.")

        frame_count = 0
        while True:
            frame = await self.queue.get()
            if frame is None:           # sentinel — stop after all frames processed
                break
            try:
                vis_frame, persons, stats = await loop.run_in_executor(
                    self._server._executor, pipeline.process_frame, frame
                )
            except Exception as e:
                logger.error(f"[{self.device_label}] Pipeline error: {e}", exc_info=True)
                continue

            await self._server._broadcast_frame(vis_frame, persons, stats, self.device_label)
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(
                    f"[{self.device_label}] frames={frame_count} | "
                    f"persons={stats['person_count']} | "
                    f"inference={stats['inference_ms']:.1f}ms | "
                    f"queue={self.queue.qsize()}"
                )

        # Sentinel received — all frames processed, notify monitors.
        logger.info(f"[{self.device_label}] All frames processed.")
        self._server._sessions.pop(self.device_label, None)
        await self._server._broadcast_video_ended(self.device_label)


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

class RecognitionServer:
    def __init__(self, host='0.0.0.0', port=3001):
        self.host = host
        self.port = port
        self.monitors: set = set()
        self.frame_count = 0
        # Shared thread pool — sessions submit blocking ML calls here.
        # max_workers=4 lets 2 cameras run nearly in parallel on GPU.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='pipeline')
        # Active sessions keyed by device_label.
        self._sessions: dict[str, _CameraSession] = {}
        # WebSocket references for live Android cameras (keyed by device_label = IP string).
        self._camera_websockets: dict = {}

    # ------------------------------------------------------------------
    # Shared broadcast helper
    # ------------------------------------------------------------------

    async def _broadcast_frame(self, vis_frame, persons, stats, device_label: str):
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
                'device':       device_label,
            },
        })

        dead = set()
        for ws in self.monitors:
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self.monitors -= dead

    async def _broadcast_video_ended(self, device_label: str):
        dead = set()
        for ws in self.monitors:
            try:
                await ws.send(json.dumps({'type': 'video_ended', 'device': device_label}))
            except Exception:
                dead.add(ws)
        self.monitors -= dead

    async def _broadcast_camera_disconnected(self, device_label: str):
        dead = set()
        for ws in self.monitors:
            try:
                await ws.send(json.dumps({'type': 'camera_disconnected', 'device': device_label}))
            except Exception:
                dead.add(ws)
        self.monitors -= dead

    async def _broadcast_error(self, msg: str):
        dead = set()
        for ws in self.monitors:
            try:
                await ws.send(json.dumps({'type': 'error', 'message': msg}))
            except Exception:
                dead.add(ws)
        self.monitors -= dead

    # ------------------------------------------------------------------
    # Connection routing
    # ------------------------------------------------------------------

    async def handle_connection(self, websocket):
        path = websocket.request.path
        if path == '/monitor':
            await self._handle_monitor(websocket)
        else:
            await self._handle_camera(websocket)

    async def _handle_monitor(self, websocket):
        """React UI client — receives broadcast results and can send video_path commands."""
        client_addr = websocket.remote_address
        logger.info(f"Monitor connected: {client_addr}")
        self.monitors.add(websocket)
        try:
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    if msg.get('type') == 'video_path':
                        path = msg.get('path', '').strip()
                        asyncio.ensure_future(self._play_video_file(path))
                    elif msg.get('type') == 'remove_camera':
                        device = msg.get('device', '')
                        if device in self._sessions:
                            self._sessions.pop(device).stop()
                        if device in self._camera_websockets:
                            ws = self._camera_websockets.pop(device)
                            await ws.close(code=1000, reason='removed_by_monitor')
                        logger.info(f"Removed device: {device}")
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.monitors.discard(websocket)
            logger.info(f"Monitor disconnected: {client_addr}")

    async def _handle_camera(self, websocket):
        """Android camera — drain WebSocket buffer fast, enqueue decoded frames."""
        client_addr = websocket.remote_address
        device_label = str(client_addr[0]) if client_addr else 'camera'
        logger.info(f"Camera connected: {client_addr}")

        # Store websocket reference so monitor can close it remotely.
        self._camera_websockets[device_label] = websocket

        # One session per device_label; if the same IP reconnects, reuse slot.
        session = _CameraSession(device_label, self)
        self._sessions[device_label] = session
        session.start()

        _MAX_FPS = 30
        _MIN_INTERVAL = 1.0 / _MAX_FPS  # seconds between accepted frames
        _last_enqueue = 0.0

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    frame_b64 = data.get('frame')
                    if not frame_b64:
                        continue
                    # FPS cap: drop frames that arrive faster than MAX_FPS
                    now = asyncio.get_event_loop().time()
                    if now - _last_enqueue < _MIN_INTERVAL:
                        continue
                    _last_enqueue = now
                    frame_bytes = base64.b64decode(frame_b64)
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        await session.queue.put(frame)
                except Exception as e:
                    logger.warning(f"[{device_label}] Frame decode error: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Camera disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"[{device_label}] Camera client error: {e}", exc_info=True)
        finally:
            session.stop()
            self._sessions.pop(device_label, None)
            self._camera_websockets.pop(device_label, None)
            await self._broadcast_camera_disconnected(device_label)
            logger.info(f"[{device_label}] Camera session cleaned up.")

    async def _play_video_file(self, video_path: str):
        """Stream a local video file through its own isolated session."""
        video_path = video_path.strip('"\'')

        if not os.path.isfile(video_path):
            await self._broadcast_error(f'File not found: {video_path}')
            logger.warning(f"Video file not found: {video_path}")
            return

        logger.info(f"Playing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await self._broadcast_error(f'Cannot open video: {video_path}')
            logger.error(f"cv2.VideoCapture failed: {video_path}")
            return

        basename = os.path.basename(video_path)
        device_label = f'video:{basename}'

        # Stop any previous session for this video path before starting a new one.
        if device_label in self._sessions:
            self._sessions.pop(device_label).stop()

        session = _CameraSession(device_label, self)
        self._sessions[device_label] = session
        session.start()

        # Pace video playback: cap at 30 FPS regardless of source FPS.
        _MAX_FPS = 30
        src_fps = cap.get(cv2.CAP_PROP_FPS) or _MAX_FPS
        frame_interval = 1.0 / min(src_fps, _MAX_FPS)
        logger.info(
            f"[{device_label}] source FPS={src_fps:.1f}, "
            f"capped to {min(src_fps, _MAX_FPS):.1f} FPS "
            f"(interval={frame_interval*1000:.1f} ms)"
        )

        loop = asyncio.get_event_loop()
        try:
            while cap.isOpened():
                t_read_start = loop.time()
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    break
                await session.queue.put(frame)
                # Sleep for the remainder of the frame interval so we don't
                # flood the queue faster than real-time playback.
                elapsed = loop.time() - t_read_start
                sleep_s = frame_interval - elapsed
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
            # Sentinel — worker drains remaining frames then exits cleanly.
            await session.queue.put(None)
        except Exception as e:
            await self._broadcast_error(f'Video read error: {e}')
            session.stop()
            self._sessions.pop(device_label, None)
        finally:
            cap.release()
            logger.info(f"Finished reading video: {video_path}")

    async def start(self):
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=20,   # send a ping every 20 s
            ping_timeout=10,    # close connection if no pong within 10 s
        ):
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
