import os
from typing import List
import numpy as np
from PIL import Image
import cv2


def make_gei_from_frames(frames: List[np.ndarray], size=(224, 224)) -> np.ndarray:
    """
    Given a list of silhouette frames (H x W, binary or float), compute GEI (average silhouette).
    Returns a float32 image normalized to [0,1] with shape (H,W).
    """
    if len(frames) == 0:
        raise ValueError('No frames provided')

    # convert frames to float and resize if needed
    proc = []
    for f in frames:
        if not isinstance(f, np.ndarray):
            f = np.array(f)
        img = f.astype(np.float32)
        # normalize to 0..1
        if img.max() > 1.0:
            img = img / 255.0
        # resize if necessary using PIL
        if img.shape != size:
            pil = Image.fromarray((img * 255).astype('uint8'))
            pil = pil.resize(size, Image.BILINEAR)
            img = np.array(pil).astype(np.float32) / 255.0
        proc.append(img)

    gei = np.mean(np.stack(proc, axis=0), axis=0)
    # clip
    gei = np.clip(gei, 0.0, 1.0).astype(np.float32)
    return gei

def process_video_to_gei(video_path: str, output_dir: str, frames_per_gei: int = 30,
                         size=(224, 224), overwrite: bool = False, return_paths: bool = True):
    """
    Process a single video file and create GEI images from every `frames_per_gei` frames.

    - `video_path`: path to mp4 file
    - `output_dir`: directory where GEI images will be saved (created if missing)
    - `frames_per_gei`: number of frames to aggregate into one GEI
    - `size`: output GEI size (H,W)
    - `overwrite`: if False and GEI file exists, skip; if True, overwrite existing GEI files
    - `return_paths`: if True, return list of written GEI paths

    GEI filenames: {video_basename}_part{N}.png (N starts at 0)
    If overwrite is True and existing GEIs are present, they will be removed first.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    # gather existing geis for this video
    pattern_pref = f"{base}_part"
    existing = [f for f in os.listdir(output_dir) if f.startswith(pattern_pref) and f.lower().endswith('.png')]
    if existing and overwrite:
        for f in existing:
            try:
                os.remove(os.path.join(output_dir, f))
            except Exception:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    frame_buf = []
    out_paths = []
    idx = 0
    part = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert to grayscale silhouette-like image by thresholding luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # simple adaptive threshold to approximate silhouette (user can replace with better bg-sub)
        _, sil = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_buf.append(sil)
        idx += 1

        if len(frame_buf) == frames_per_gei:
            out_name = f"{base}_part{part}.png"
            out_path = os.path.join(output_dir, out_name)
            if os.path.exists(out_path) and not overwrite:
                out_paths.append(out_path)
            else:
                gei = make_gei_from_frames(frame_buf, size=size)
                Image.fromarray((gei * 255).astype('uint8')).save(out_path)
                out_paths.append(out_path)
            part += 1
            frame_buf = []

    # handle leftover frames (if any) as final part
    if len(frame_buf) > 0:
        out_name = f"{base}_part{part}.png"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path) and not overwrite:
            out_paths.append(out_path)
        else:
            gei = make_gei_from_frames(frame_buf, size=size)
            Image.fromarray((gei * 255).astype('uint8')).save(out_path)
            out_paths.append(out_path)

    cap.release()
    if return_paths:
        return out_paths
    return []
