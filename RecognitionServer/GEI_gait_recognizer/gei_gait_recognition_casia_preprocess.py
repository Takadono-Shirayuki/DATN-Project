"""
CASIA-B Preprocessing Pipeline
Bước 1: Tạo GEI (Gait Energy Image) từ các frame ảnh thô của CASIA-B
Bước 2: Resize GEI về 224x224 (GPU-accelerated)

Chạy từ bất kỳ thư mục nào - script tự động xác định đường dẫn theo vị trí file.
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================
# Bước 1: Tạo GEI từ CASIA-B
# =============================================================

def find_frame_folders(root):
    """Tìm tất cả thư mục lá chứa ảnh frame."""
    frame_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirnames:
            continue  # chỉ lấy thư mục lá
        if (all(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
                for f in filenames) and len(filenames) > 0):
            frame_folders.append(dirpath)
    return frame_folders


def prepare_gei_tasks(frame_folders, num_frames_per_gei, casia_b_root, output_root):
    """Chuẩn bị danh sách task tạo GEI."""
    tasks = []
    for folder in frame_folders:
        frame_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        if len(frame_files) < num_frames_per_gei:
            continue
        rel_path = os.path.relpath(folder, casia_b_root)
        label = rel_path.split(os.sep)[0]
        label_out_dir = os.path.join(output_root, label)
        os.makedirs(label_out_dir, exist_ok=True)
        for i in range(0, len(frame_files) - num_frames_per_gei + 1, num_frames_per_gei):
            batch = frame_files[i:i + num_frames_per_gei]
            batch_paths = [os.path.join(folder, f) for f in batch]
            folder_name = os.path.basename(folder)
            out_name = f"{folder_name}_gei_{i // num_frames_per_gei + 1}.png"
            out_path = os.path.join(label_out_dir, out_name)
            tasks.append((batch_paths, out_path))
    return tasks


def load_gei_images(batch_paths):
    """Đọc một batch frames dưới dạng grayscale tensor."""
    imgs_np = [cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in batch_paths]
    imgs_np = np.stack(imgs_np) / 255.0  # (num_frames, H, W)
    return torch.from_numpy(imgs_np)


def create_gei_on_gpu(imgs_tensor, device):
    """Tính GEI (trung bình frame) trên GPU."""
    imgs_tensor = imgs_tensor.to(device, non_blocking=True)
    return imgs_tensor.mean(dim=0)


def run_create_gei(casia_b_root, output_root, num_frames_per_gei=30, batch_size=64, device=None):
    """
    Tạo GEI từ CASIA-B frame folders.
    Args:
        casia_b_root: Thư mục gốc CASIA-B
        output_root:  Thư mục lưu GEI đầu ra
        num_frames_per_gei: Số frame mỗi GEI (mặc định 30)
        batch_size:   Số GEI xử lý mỗi batch GPU
        device:       torch.device (mặc định auto-detect)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_root, exist_ok=True)
    frame_folders = find_frame_folders(casia_b_root)
    print(f"[Bước 1] Tìm thấy {len(frame_folders)} thư mục chuỗi frames. Device: {device}")

    tasks = prepare_gei_tasks(frame_folders, num_frames_per_gei, casia_b_root, output_root)
    to_pil = transforms.ToPILImage()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_gei_images, batch_paths) for batch_paths, _ in tasks]

        for i in tqdm(range(0, len(futures), batch_size), desc="Tạo GEI trên GPU"):
            batch_futures = futures[i:i + batch_size]
            batch_out_paths = [tasks[i + j][1] for j in range(len(batch_futures))]

            imgs_tensors = [f.result() for f in batch_futures]
            imgs_batch = torch.stack(imgs_tensors)  # (B, num_frames, H, W)

            geis = imgs_batch.to(device, non_blocking=True).mean(dim=1)

            for gei_tensor, out_path in zip(geis, batch_out_paths):
                gei_img = to_pil(gei_tensor.cpu().clamp(0, 1))
                gei_img.save(out_path)

            del imgs_batch, geis, imgs_tensors
            torch.cuda.empty_cache()

    print(f"[Bước 1] Hoàn thành. Đã lưu GEI vào: {output_root}")


# =============================================================
# Bước 2: Resize GEI về 224x224
# =============================================================

def prepare_resize_tasks(source_root, dest_root):
    """Chuẩn bị danh sách task resize ảnh GEI."""
    import shutil
    
    labels = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]

    resize_tasks = []
    for label in labels:
        label_src = os.path.join(source_root, label)
        label_dst = os.path.join(dest_root, label)
        
        # Xóa thư mục label cũ nếu tồn tại để tránh dữ liệu thừa
        if os.path.exists(label_dst):
            shutil.rmtree(label_dst)
        
        os.makedirs(label_dst, exist_ok=True)
        for root, _, files in os.walk(label_src):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    src_img_path = os.path.join(root, file)
                    dst_img_path = os.path.join(label_dst, file)
                    resize_tasks.append((src_img_path, dst_img_path))
    return resize_tasks


def load_image_rgb(path):
    """Đọc ảnh màu (RGB) dưới dạng float tensor."""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)       # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # RGB
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C,H,W)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        img = torch.zeros(3, 64, 64)
    return img


def save_tensor_as_image(tensor, path):
    """Lưu tensor (C,H,W) thành file ảnh."""
    img = transforms.ToPILImage()(tensor.cpu().clamp(0, 1))
    img.save(path)


def run_resize_gei(source_root, dest_root, resize_shape=(224, 224), batch_size=512, device=None):
    """
    Resize toàn bộ ảnh GEI về kích thước resize_shape.
    Args:
        source_root:  Thư mục GEI nguồn (đầu ra Bước 1)
        dest_root:    Thư mục GEI đích (224x224)
        resize_shape: Kích thước đích (H, W)
        batch_size:   Số ảnh mỗi batch GPU
        device:       torch.device (mặc định auto-detect)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resize_tasks = prepare_resize_tasks(source_root, dest_root)
    print(f"[Bước 2] Tổng số ảnh cần resize: {len(resize_tasks)}. Device: {device}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_image_rgb, src) for src, _ in resize_tasks]

        for i in tqdm(range(0, len(futures), batch_size), desc="Batch resizing với GPU"):
            batch_futures = futures[i:i + batch_size]
            batch_out_paths = [resize_tasks[i + j][1] for j in range(len(batch_futures))]

            imgs_tensors = [f.result() for f in batch_futures]
            imgs_batch = torch.stack(imgs_tensors)  # (B,C,H,W)

            imgs_resized = F.interpolate(
                imgs_batch.to(device, non_blocking=True),
                size=resize_shape,
                mode='bilinear',
                align_corners=False,
            )

            for img_tensor, out_path in zip(imgs_resized, batch_out_paths):
                save_tensor_as_image(img_tensor, out_path)

            del imgs_batch, imgs_resized, imgs_tensors
            torch.cuda.empty_cache()

    print(f"[Bước 2] Hoàn thành. Đã lưu GEI 224x224 vào: {dest_root}")


# =============================================================
# Main: chạy Pipeline 2 bước
# =============================================================
if __name__ == "__main__":
    import shutil

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    casia_b_root  = os.path.join(_HERE, '..', '..', 'dataset', 'CASIA-B')
    gei_raw_root  = os.path.join(_HERE, '_gei_tmp')          # thư mục tạm, sẽ xóa sau
    gei_224_root  = os.path.join(_HERE, 'dataset_processed') # thư mục đích 224x224

    print("=== Pipeline Tiền xử lý CASIA-B ===")
    print(f"  CASIA-B  : {os.path.normpath(casia_b_root)}")
    print(f"  GEI tmp  : {os.path.normpath(gei_raw_root)}")
    print(f"  GEI 224  : {os.path.normpath(gei_224_root)}")
    print(f"  Device   : {device}\n")

    try:
        # Bước 1: Tạo GEI thô vào thư mục tạm
        run_create_gei(
            casia_b_root=casia_b_root,
            output_root=gei_raw_root,
            num_frames_per_gei=30,
            batch_size=64,
            device=device,
        )

        # Bước 2: Resize GEI → 224x224
        run_resize_gei(
            source_root=gei_raw_root,
            dest_root=gei_224_root,
            resize_shape=(224, 224),
            batch_size=512,
            device=device,
        )
    finally:
        if os.path.exists(gei_raw_root):
            shutil.rmtree(gei_raw_root)
            print(f"[Cleanup] Đã xóa thư mục tạm: {os.path.normpath(gei_raw_root)}")

    print("\n=== Hoàn thành toàn bộ pipeline ===")
