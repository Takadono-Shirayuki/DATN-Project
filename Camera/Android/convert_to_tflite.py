#!/usr/bin/env python3
"""
Convert YOLO models to TensorFlow Lite format for Android
Usage: python convert_to_tflite.py --input yolov8n.pt --output app/src/main/assets/
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def convert_to_tflite(input_model: str, output_dir: str):
    """Convert YOLO PyTorch model to TFLite format with quantization"""
    
    print(f"[*] Loading model: {input_model}")
    model = YOLO(input_model)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Exporting to TFLite format...")
    # Export to quantized TFLite
    results = model.export(
        format='tflite',
        imgsz=640,
        int8=True,  # Quantize to int8
        data=None,
        device='cpu',
        half=False,
        keras=False,
        optimize=True,
        dynamic=False,
        simplify=True,
        opset=12,
        workspace=4,
        nms=False,
        agnostic_nms=False,
        topk_per_class=100,
        topk_all=100,
        iou_thres=0.45,
        conf_thres=0.25,
        verbose=False
    )
    
    # Copy to output directory
    tflite_path = str(results)
    output_path = os.path.join(output_dir, Path(input_model).stem + '.tflite')
    
    print(f"[*] Copying: {tflite_path} -> {output_path}")
    import shutil
    shutil.copy(tflite_path, output_path)
    
    # Also create quantized version
    quant_output = os.path.join(output_dir, Path(input_model).stem + '_quant.tflite')
    print(f"[+] Model saved to: {output_path}")
    print(f"[+] File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"[*] Copy this file to: app/src/main/assets/")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO to TFLite')
    parser.add_argument('--input', required=True, help='Input YOLO model (e.g., yolov8n.pt)')
    parser.add_argument('--output', default='./assets', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[!] Model not found: {args.input}")
        return
    
    convert_to_tflite(args.input, args.output)


if __name__ == '__main__':
    main()
