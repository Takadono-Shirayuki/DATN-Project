"""
Script to organize raw dataset files into walking and non-walking categories
"""
import os
import shutil
from pathlib import Path

# Define paths
raw_dataset_dir = Path('raw dataset')
walking_dir = raw_dataset_dir / 'walking'
non_walking_dir = raw_dataset_dir / 'non-walking'

# Ensure target directories exist
walking_dir.mkdir(exist_ok=True)
non_walking_dir.mkdir(exist_ok=True)

moved_files = {'walking': 0, 'non-walking': 0}

print("="*70)
print("Organizing Raw Dataset Files")
print("="*70)

# 1. Move all .avi files directly in raw dataset/ to walking/
print("\n[1] Moving files from root directory to walking/...")
for file in raw_dataset_dir.glob('*.avi'):
    if file.is_file():
        dest = walking_dir / file.name
        if not dest.exists():
            shutil.move(str(file), str(dest))
            moved_files['walking'] += 1
            print(f"  ✓ {file.name} → walking/")

# 2. Process hmdb51 subdirectories
hmdb51_dir = raw_dataset_dir / 'hmdb51'
if hmdb51_dir.exists():
    print("\n[2] Processing hmdb51/...")
    for split_dir in hmdb51_dir.iterdir():
        if split_dir.is_dir() and split_dir.name in ['train', 'test', 'val']:
            # Process walking subdirectory
            walking_subdir = split_dir / 'walking'
            if walking_subdir.exists():
                for file in walking_subdir.glob('*'):
                    if file.is_file():
                        dest = walking_dir / file.name
                        # Avoid name collision
                        if dest.exists():
                            base_name = file.stem
                            ext = file.suffix
                            counter = 1
                            while dest.exists():
                                dest = walking_dir / f"{base_name}_{counter}{ext}"
                                counter += 1
                        shutil.move(str(file), str(dest))
                        moved_files['walking'] += 1
                        print(f"  ✓ hmdb51/{split_dir.name}/walking/{file.name} → walking/{dest.name}")
            
            # Process non_walking subdirectory
            non_walking_subdir = split_dir / 'non_walking'
            if non_walking_subdir.exists():
                for file in non_walking_subdir.glob('*'):
                    if file.is_file():
                        dest = non_walking_dir / file.name
                        # Avoid name collision
                        if dest.exists():
                            base_name = file.stem
                            ext = file.suffix
                            counter = 1
                            while dest.exists():
                                dest = non_walking_dir / f"{base_name}_{counter}{ext}"
                                counter += 1
                        shutil.move(str(file), str(dest))
                        moved_files['non-walking'] += 1
                        print(f"  ✓ hmdb51/{split_dir.name}/non_walking/{file.name} → non-walking/{dest.name}")

# 3. Process other_dataset subdirectories
other_dataset_dir = raw_dataset_dir / 'other_dataset'
if other_dataset_dir.exists():
    print("\n[3] Processing other_dataset/...")
    for split_dir in other_dataset_dir.iterdir():
        if split_dir.is_dir() and split_dir.name in ['train', 'test', 'val']:
            # Process walking subdirectory
            walking_subdir = split_dir / 'walking'
            if walking_subdir.exists():
                for file in walking_subdir.glob('*'):
                    if file.is_file():
                        dest = walking_dir / file.name
                        # Avoid name collision
                        if dest.exists():
                            base_name = file.stem
                            ext = file.suffix
                            counter = 1
                            while dest.exists():
                                dest = walking_dir / f"{base_name}_{counter}{ext}"
                                counter += 1
                        shutil.move(str(file), str(dest))
                        moved_files['walking'] += 1
                        print(f"  ✓ other_dataset/{split_dir.name}/walking/{file.name} → walking/{dest.name}")
            
            # Process non_walking subdirectory
            non_walking_subdir = split_dir / 'non_walking'
            if non_walking_subdir.exists():
                for file in non_walking_subdir.glob('*'):
                    if file.is_file():
                        dest = non_walking_dir / file.name
                        # Avoid name collision
                        if dest.exists():
                            base_name = file.stem
                            ext = file.suffix
                            counter = 1
                            while dest.exists():
                                dest = non_walking_dir / f"{base_name}_{counter}{ext}"
                                counter += 1
                        shutil.move(str(file), str(dest))
                        moved_files['non-walking'] += 1
                        print(f"  ✓ other_dataset/{split_dir.name}/non_walking/{file.name} → non-walking/{dest.name}")

# 4. Clean up empty subdirectories
print("\n[4] Cleaning up empty directories...")
for subdir in [hmdb51_dir, other_dataset_dir]:
    if subdir.exists():
        for split_dir in subdir.iterdir():
            if split_dir.is_dir():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir() and not any(category_dir.iterdir()):
                        category_dir.rmdir()
                        print(f"  ✓ Removed empty: {category_dir.relative_to(raw_dataset_dir)}")
                if not any(split_dir.iterdir()):
                    split_dir.rmdir()
                    print(f"  ✓ Removed empty: {split_dir.relative_to(raw_dataset_dir)}")
        if not any(subdir.iterdir()):
            subdir.rmdir()
            print(f"  ✓ Removed empty: {subdir.relative_to(raw_dataset_dir)}")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"Walking files moved: {moved_files['walking']}")
print(f"Non-walking files moved: {moved_files['non-walking']}")
print(f"Total files organized: {sum(moved_files.values())}")
print("\n✅ Dataset organization complete!")
print(f"  - Walking videos: raw dataset/walking/")
print(f"  - Non-walking videos: raw dataset/non-walking/")
