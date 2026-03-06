#!/usr/bin/env python3
"""
Gait Recognition - Camera App (Linux/Desktop)

Modes:
  - streaming: Send raw camera feed to server
  - recognition: Process with YOLO + Action Classifier + GEI before sending
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from camera_capture import CameraCapture
from network_client import NetworkClient
from yolo_detector import YOLODetector
import cv2
import numpy as np
import time


class CameraApp:
    def __init__(self, server_ip, server_port=5000, mode='streaming'):
        self.server_ip = server_ip
        self.server_port = server_port
        self.mode = mode
        
        self.camera = None
        self.client = None
        self.detector = None
        
        self.setup()
    
    def setup(self):
        """Initialize camera and network"""
        print(f"[{self.mode.upper()}] Initializing...")
        
        # Initialize camera
        self.camera = CameraCapture(camera_id=0, frame_width=640, frame_height=480, fps=30)
        self.camera.start()
        
        if not self.camera.is_available():
            print("ERROR: Camera not available!")
            sys.exit(1)
        
        # Initialize network client
        self.client = NetworkClient(self.server_ip, self.server_port)
        if not self.client.connect():
            print(f"ERROR: Cannot connect to server {self.server_ip}:{self.server_port}")
            sys.exit(1)
        
        # Register device
        self.client.send_registration(f"camera-{self.mode}", device_type=self.mode)
        
        # Load models if recognition mode
        if self.mode == 'recognition':
            print("Loading YOLO models...")
            try:
                self.detector = YOLODetector()
                print("Models loaded successfully")
            except Exception as e:
                print(f"ERROR loading models: {e}")
                sys.exit(1)
    
    def run_streaming_mode(self):
        """Run camera in streaming mode - send raw frames"""
        print("[STREAMING] Started. Press 'q' to quit.")
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                # Display frame
                cv2.imshow('Camera - Streaming Mode', frame)
                
                # Send to server
                metadata = {
                    'mode': 'streaming',
                    'timestamp': time.time()
                }
                self.client.send_frame(frame, metadata)
                
                frame_count += 1
                
                # Print FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"[STREAMING] Frames: {frame_count}, FPS: {fps:.1f}")
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cv2.destroyAllWindows()
    
    def run_recognition_mode(self):
        """Run camera in recognition mode - process with AI models"""
        print("[RECOGNITION] Started. Press 'q' to quit.")
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                infer_start = time.time()

                # Detect persons
                detections = self.detector.detect_persons(frame)
                
                # Estimate pose
                pose_results = self.detector.estimate_pose(frame)

                # Extract silhouette image (white foreground on black background)
                silhouette_frame, person_count = self.detector.extract_person_silhouette(frame, detections)
                
                # Draw annotations
                annotated_frame = self.detector.draw_detections(frame, detections)

                infer_ms = (time.time() - infer_start) * 1000.0
                
                # Display side-by-side: detections (left), silhouette (right)
                preview = np.hstack((annotated_frame, silhouette_frame))
                cv2.imshow('Camera - Recognition Mode (Detection | Silhouette)', preview)
                
                # Send silhouette to server, not raw live frame.
                pose_count = 0
                if pose_results and getattr(pose_results[0], 'keypoints', None) is not None:
                    pose_count = int(pose_results[0].keypoints.data.shape[0])

                metadata = {
                    'mode': 'recognition',
                    'detections': int(person_count),
                    'pose_persons': pose_count,
                    'timestamp': time.time(),
                    'action': 'unknown',  # TODO: connect action classifier checkpoint
                    'confidence': 0.0,
                    'pipeline': 'yolo_detection+pose+silhouette',
                    'model_inference_ms': round(infer_ms, 2),
                }
                self.client.send_frame(silhouette_frame, metadata)
                
                frame_count += 1
                
                # Print FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(
                        f"[RECOGNITION] Frames: {frame_count}, FPS: {fps:.1f}, "
                        f"persons: {person_count}, infer: {infer_ms:.1f}ms"
                    )
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cv2.destroyAllWindows()
    
    def run(self):
        """Start the app"""
        try:
            if self.mode == 'streaming':
                self.run_streaming_mode()
            elif self.mode == 'recognition':
                self.run_recognition_mode()
            else:
                print(f"ERROR: Unknown mode '{self.mode}'")
                sys.exit(1)
        
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        if self.camera:
            self.camera.stop()
        if self.client:
            self.client.disconnect()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='GR2 Camera App - Send camera feed to server'
    )
    parser.add_argument(
        '--mode',
        choices=['streaming', 'recognition'],
        default='streaming',
        help='Operating mode'
    )
    parser.add_argument(
        '--server-ip',
        required=True,
        help='Server IP address'
    )
    parser.add_argument(
        '--server-port',
        type=int,
        default=3000,
        help='Server port (default: 3000)'
    )
    
    args = parser.parse_args()
    
    app = CameraApp(
        server_ip=args.server_ip,
        server_port=args.server_port,
        mode=args.mode
    )
    app.run()


if __name__ == '__main__':
    main()
