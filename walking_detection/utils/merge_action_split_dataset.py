"""
Script để đẩy dữ liệu từ dataset_action_split vào dataset
- Map Walking -> walking
- Map các action khác -> non_walking  
- Chia tỉ lệ train/val/test hợp lý (70/15/15)
- Cân bằng tỉ lệ walking/non_walking trong mỗi split
- Convert .avi sang .mp4
"""
import os
import shutil
import random
import subprocess
from pathlib import Path
from collections import defaultdict

# Cấu hình
SOURCE_DIR = Path(__file__).parent.parent / "dataset_action_split"
TARGET_DIR = Path(__file__).parent.parent / "dataset"

# Tỉ lệ chia train/val/test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Mapping action -> class
WALKING_ACTION = "Walking"
NON_WALKING_ACTIONS = ["Fall Down", "Sit down", "Lying Down", "Stand up", "Standing", "Sitting"]

random.seed(42)  # Để reproducible


def get_video_files(directory):
    """Lấy tất cả các file video từ thư mục"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    if not os.path.exists(directory):
        return video_files
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))
    
    return sorted(video_files)


def collect_videos_from_action_split():
    """Thu thập tất cả videos từ dataset_action_split và phân loại"""
    videos = {
        'walking': [],
        'non_walking': []
    }
    
    splits = ['train', 'test']
    
    for split in splits:
        split_dir = SOURCE_DIR / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} không tồn tại")
            continue
        
        # Lấy videos từ Walking -> walking
        walking_dir = split_dir / WALKING_ACTION
        if walking_dir.exists():
            walking_videos = get_video_files(str(walking_dir))
            for video_path in walking_videos:
                videos['walking'].append({
                    'source': video_path,
                    'original_split': split,
                    'filename': os.path.basename(video_path)
                })
        
        # Lấy videos từ các action khác -> non_walking
        for action in NON_WALKING_ACTIONS:
            action_dir = split_dir / action
            if action_dir.exists():
                action_videos = get_video_files(str(action_dir))
                for video_path in action_videos:
                    videos['non_walking'].append({
                        'source': video_path,
                        'original_split': split,
                        'filename': os.path.basename(video_path)
                    })
    
    return videos


def create_split_plan(videos, balance_classes=True):
    """
    Tạo kế hoạch chia dataset thành train/val/test
    Nếu balance_classes=True, sẽ cân bằng số lượng walking và non_walking
    """
    plan = {
        'train': {'walking': [], 'non_walking': []},
        'val': {'walking': [], 'non_walking': []},
        'test': {'walking': [], 'non_walking': []}
    }
    
    # Shuffle để tránh bias
    random.shuffle(videos['walking'])
    random.shuffle(videos['non_walking'])
    
    # Quyết định số lượng mỗi class dựa trên class nhỏ hơn
    if balance_classes:
        # Lấy số lượng class nhỏ hơn làm chuẩn
        min_count = min(len(videos['walking']), len(videos['non_walking']))
        
        # Giới hạn cả 2 class về cùng số lượng
        walking_selected = videos['walking'][:min_count]
        non_walking_selected = videos['non_walking'][:min_count]
        
        print(f"  Cân bằng: Walking={len(videos['walking'])} -> {len(walking_selected)}, "
              f"Non-walking={len(videos['non_walking'])} -> {len(non_walking_selected)}")
    else:
        # Dùng tất cả videos
        walking_selected = videos['walking']
        non_walking_selected = videos['non_walking']
    
    # Chia train/val/test cho từng class
    for class_name, selected_videos in [('walking', walking_selected), 
                                         ('non_walking', non_walking_selected)]:
        total = len(selected_videos)
        
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        test_count = total - train_count - val_count  # Phần còn lại vào test
        
        # Chia videos
        plan['train'][class_name] = selected_videos[:train_count]
        plan['val'][class_name] = selected_videos[train_count:train_count + val_count]
        plan['test'][class_name] = selected_videos[train_count + val_count:]
    
    return plan


def print_statistics(videos, plan):
    """In thống kê về dataset"""
    print("\n" + "="*70)
    print("THỐNG KÊ DATASET")
    print("="*70)
    
    print("\n📊 Videos từ dataset_action_split:")
    print(f"  Walking: {len(videos['walking'])} videos")
    print(f"  Non-walking: {len(videos['non_walking'])} videos")
    print(f"  Tổng: {len(videos['walking']) + len(videos['non_walking'])} videos")
    
    print("\n📦 Kế hoạch chia dataset:")
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-walking':<15} {'Total':<15} {'Ratio':<15}")
    print("-"*75)
    
    for split in ['train', 'val', 'test']:
        walking_count = len(plan[split]['walking'])
        non_walking_count = len(plan[split]['non_walking'])
        total = walking_count + non_walking_count
        total_all = sum(len(plan[s]['walking']) + len(plan[s]['non_walking']) 
                       for s in ['train', 'val', 'test'])
        ratio = (total / total_all * 100) if total_all > 0 else 0
        
        print(f"{split:<15} {walking_count:<15} {non_walking_count:<15} "
              f"{total:<15} {ratio:>5.1f}%")
    
    print("="*70)


def convert_video_to_mp4(input_path, output_path):
    """
    Convert video từ .avi sang .mp4 bằng ffmpeg
    """
    try:
        # Sử dụng ffmpeg để convert
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',  # H.264 codec
            '-c:a', 'aac',      # AAC audio
            '-preset', 'medium', # Balance giữa speed và quality
            '-crf', '23',       # Quality (18-28, 23 là good default)
            '-y',               # Overwrite output file
            str(output_path)
        ]
        
        # Chạy ffmpeg và ẩn output
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        return False
    except FileNotFoundError:
        print("\n⚠️  ffmpeg không được tìm thấy. Vui lòng cài đặt ffmpeg.")
        return False


def copy_videos(plan, dry_run=True):
    """
    Copy và convert videos từ source sang target theo kế hoạch
    Convert .avi sang .mp4
    """
    if dry_run:
        print("\n[DRY RUN] Chế độ test - không copy file thực sự")
    else:
        print("\n🔄 Bắt đầu copy và convert files...")
    
    # Đếm tổng số file sẽ copy
    total_files = sum(
        len(plan[split][class_name])
        for split in ['train', 'val', 'test']
        for class_name in ['walking', 'non_walking']
    )
    
    copied = 0
    errors = []
    
    for split in ['train', 'val', 'test']:
        for class_name in ['walking', 'non_walking']:
            target_dir = TARGET_DIR / split / class_name
            if not dry_run:
                os.makedirs(target_dir, exist_ok=True)
            
            for video_info in plan[split][class_name]:
                source = video_info['source']
                source_filename = video_info['filename']
                
                # Đổi extension từ .avi sang .mp4
                base_name, ext = os.path.splitext(source_filename)
                if ext.lower() == '.avi':
                    target_filename = f"{base_name}.mp4"
                else:
                    target_filename = source_filename
                
                target = target_dir / target_filename
                
                if dry_run:
                    print(f"  [DRY] {source} -> {target}")
                    copied += 1  # Đếm trong dry run
                else:
                    try:
                        # Nếu file đã tồn tại, skip hoặc rename
                        if target.exists():
                            # Thêm prefix để tránh trùng
                            base, ext = os.path.splitext(target_filename)
                            new_filename = f"{base}_from_split{ext}"
                            target = target_dir / new_filename
                            target_filename = new_filename
                        
                        # Convert nếu là .avi, nếu không thì copy trực tiếp
                        if source.lower().endswith('.avi'):
                            if not convert_video_to_mp4(source, target):
                                errors.append((source, "Convert failed"))
                                continue
                        else:
                            shutil.copy2(source, target)
                        
                        copied += 1
                        if copied % 50 == 0:
                            print(f"  Đã xử lý {copied}/{total_files} files... ({copied*100//total_files}%)")
                    except Exception as e:
                        errors.append((source, str(e)))
    
    if errors:
        print(f"\n⚠️  Có {len(errors)} lỗi khi copy/convert:")
        for source, error in errors[:10]:
            print(f"    {os.path.basename(source)}: {error}")
        if len(errors) > 10:
            print(f"    ... và {len(errors) - 10} lỗi khác")
    
    return copied, errors


def main():
    """Hàm chính"""
    print("="*70)
    print("MERGE DATASET_ACTION_SPLIT VÀO DATASET")
    print("="*70)
    
    # Kiểm tra thư mục source
    if not SOURCE_DIR.exists():
        print(f"\n❌ Lỗi: Không tìm thấy {SOURCE_DIR}")
        return
    
    # Kiểm tra thư mục target
    if not TARGET_DIR.exists():
        print(f"\n⚠️  Thư mục {TARGET_DIR} chưa tồn tại, sẽ tạo mới...")
        os.makedirs(TARGET_DIR, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for class_name in ['walking', 'non_walking']:
                os.makedirs(TARGET_DIR / split / class_name, exist_ok=True)
    
    print(f"\n📁 Source: {SOURCE_DIR}")
    print(f"📁 Target: {TARGET_DIR}")
    
    # Bước 1: Thu thập videos
    print("\n📥 Đang thu thập videos từ dataset_action_split...")
    videos = collect_videos_from_action_split()
    
    if len(videos['walking']) == 0 and len(videos['non_walking']) == 0:
        print("\n❌ Không tìm thấy video nào!")
        return
    
    # Bước 2: Tạo kế hoạch chia
    print("\n📋 Đang tạo kế hoạch chia dataset...")
    plan = create_split_plan(videos, balance_classes=True)
    
    # Bước 3: In thống kê
    print_statistics(videos, plan)
    
    # Bước 4: Dry run
    print("\n" + "="*70)
    print("DRY RUN - KIỂM TRA")
    print("="*70)
    
    # Đếm tổng số file
    total_to_copy = sum(
        len(plan[split][class_name])
        for split in ['train', 'val', 'test']
        for class_name in ['walking', 'non_walking']
    )
    
    copied_dry, errors_dry = copy_videos(plan, dry_run=True)
    
    # Xác nhận
    print(f"\n⚠️  Sẽ copy và convert {copied_dry} files vào dataset (từ .avi sang .mp4).")
    print(f"    Lưu ý: Quá trình convert có thể mất nhiều thời gian.")
    response = input("Tiếp tục? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Đã hủy.")
        return
    
    # Bước 5: Copy thực sự
    print("\n" + "="*70)
    print("COPY FILES")
    print("="*70)
    copied, errors = copy_videos(plan, dry_run=False)
    
    print(f"\n✅ Hoàn thành! Đã copy {copied} files.")
    if errors:
        print(f"⚠️  Có {len(errors)} lỗi (xem chi tiết ở trên).")
    
    # Thống kê cuối cùng
    print("\n" + "="*70)
    print("THỐNG KÊ CUỐI CÙNG")
    print("="*70)
    
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-walking':<15} {'Total':<15}")
    print("-"*60)
    
    for split in ['train', 'val', 'test']:
        walking_dir = TARGET_DIR / split / 'walking'
        non_walking_dir = TARGET_DIR / split / 'non_walking'
        
        walking_count = len(get_video_files(str(walking_dir)))
        non_walking_count = len(get_video_files(str(non_walking_dir)))
        total = walking_count + non_walking_count
        
        print(f"{split:<15} {walking_count:<15} {non_walking_count:<15} {total:<15}")
    
    print("="*70)


if __name__ == "__main__":
    main()
