import os
import time
import cv2
import json

class Recorder:
    def __init__(self, out_dir='person_videos', fps=25.0, frame_size=(224,224), codec='mp4v', timeout=5.0):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.frame_size = tuple(frame_size)
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.timeout = float(timeout)
        self._writers = {}  # pid -> { writer, start_ts, last_seen, path, keypoints_log }

    def _make_path(self, pid):
        ts = int(time.time())
        safe_pid = str(pid).replace(os.path.sep, '_')
        return os.path.join(self.out_dir, f'person_{safe_pid}_{ts}.mp4')

    def _create_writer(self, pid):
        path = self._make_path(pid)
        writer = cv2.VideoWriter(path, self.fourcc, self.fps, self.frame_size)
        self._writers[pid] = {
            'writer': writer, 
            'start_ts': time.time(), 
            'last_seen': time.time(), 
            'path': path,
            'keypoints_log': []  # List of keypoints per frame
        }
        return writer

    def update(self, pid, frame, keypoints=None):
        """
        pid: identifier (string/int)
        frame: BGR numpy array with shape matching frame_size (or will be resized)
        keypoints: list of [x, y, confidence] for pose keypoints (optional)
        """
        if frame is None:
            return
        # ensure frame size
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        info = self._writers.get(pid)
        if info is None:
            writer = self._create_writer(pid)
            info = self._writers[pid]
        else:
            writer = info['writer']
            info['last_seen'] = time.time()
        writer.write(frame)
        
        # Log keypoints if provided
        if keypoints is not None:
            info['keypoints_log'].append(keypoints)
        
        self._cleanup_stale()

    def _cleanup_stale(self):
        now = time.time()
        stale = [pid for pid, v in self._writers.items() if now - v['last_seen'] > self.timeout]
        for pid in stale:
            self._close_writer(pid)

    def _close_writer(self, pid):
        info = self._writers.pop(pid, None)
        if info:
            w = info['writer']
            try:
                w.release()
            except Exception:
                pass
            
            # Save keypoints to JSON file with same name as video
            if info['keypoints_log']:
                video_path = info['path']
                json_path = os.path.splitext(video_path)[0] + '.json'
                try:
                    with open(json_path, 'w') as f:
                        json.dump({
                            'fps': self.fps,
                            'frame_size': self.frame_size,
                            'frames': info['keypoints_log']
                        }, f, indent=2)
                except Exception:
                    pass

    def close_all(self):
        pids = list(self._writers.keys())
        for pid in pids:
            self._close_writer(pid)
