"""
Dataset Balancing Script
1. First removes all "Meet and Split_Meet and Split" videos (dry run first)
2. Then balances walking and non_walking videos (using walking as standard)
3. Optionally redistributes dataset to 70-15-15 train-val-test ratio
"""
import os
import glob
import shutil
from pathlib import Path
from collections import defaultdict
import random


def get_video_files(directory):
    """Get all video files from a directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    if not os.path.exists(directory):
        return video_files
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))
    
    return sorted(video_files)


def find_videos_by_pattern(dataset_dir, pattern):
    """Find all videos matching a pattern across all splits and classes"""
    matches = defaultdict(lambda: defaultdict(list))
    
    splits = ['train', 'test', 'val']
    classes = ['walking', 'non_walking']
    
    for split in splits:
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, split, class_name)
            if not os.path.exists(class_dir):
                continue
            
            video_files = get_video_files(class_dir)
            for video_path in video_files:
                filename = os.path.basename(video_path)
                # Match files that contain the pattern (handles both with and without prefix)
                if pattern in filename:
                    matches[split][class_name].append(video_path)
    
    return matches


def count_videos_by_pattern(dataset_dir, pattern):
    """Count videos matching a pattern"""
    matches = find_videos_by_pattern(dataset_dir, pattern)
    total = 0
    breakdown = {}
    
    for split in matches:
        split_total = 0
        for class_name in matches[split]:
            count = len(matches[split][class_name])
            split_total += count
            total += count
        breakdown[split] = split_total
    
    return total, breakdown, matches


def delete_videos(video_paths, dry_run=True):
    """Delete videos, returns count of deleted files"""
    deleted = 0
    errors = []
    
    for video_path in video_paths:
        if dry_run:
            print(f"  [DRY RUN] Would delete: {video_path}")
        else:
            try:
                os.remove(video_path)
                deleted += 1
            except Exception as e:
                errors.append((video_path, str(e)))
    
    if errors:
        print(f"\n  Errors encountered:")
        for path, error in errors[:10]:  # Show first 10 errors
            print(f"    {os.path.basename(path)}: {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")
    
    return deleted


def get_dataset_counts(dataset_dir):
    """Get current video counts for each split and class"""
    counts = {}
    splits = ['train', 'test', 'val']
    classes = ['walking', 'non_walking']
    
    for split in splits:
        counts[split] = {}
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, split, class_name)
            videos = get_video_files(class_dir)
            counts[split][class_name] = len(videos)
    
    return counts


def redistribute_dataset_701515(dataset_dir, dry_run=True):
    """
    Redistribute dataset to 70-15-15 train-val-test ratio
    Collects all videos, shuffles them, and redistributes maintaining class balance
    """
    print("\n" + "="*70)
    print("REDISTRIBUTING DATASET TO 70-15-15 RATIO")
    print("="*70)
    
    classes = ['walking', 'non_walking']
    splits = ['train', 'test', 'val']
    target_ratios = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    
    # Step 1: Collect all videos by class
    all_videos = {'walking': [], 'non_walking': []}
    
    print("\nCollecting all videos from current splits...")
    for split in splits:
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, split, class_name)
            if os.path.exists(class_dir):
                videos = get_video_files(class_dir)
                for video_path in videos:
                    all_videos[class_name].append({
                        'path': video_path,
                        'current_split': split,
                        'filename': os.path.basename(video_path)
                    })
    
    print(f"\nTotal videos collected:")
    for class_name in classes:
        print(f"  {class_name}: {len(all_videos[class_name])} videos")
    
    # Step 2: Shuffle and redistribute
    random.seed(42)  # For reproducibility
    
    redistribution_plan = {'train': {'walking': [], 'non_walking': []},
                          'val': {'walking': [], 'non_walking': []},
                          'test': {'walking': [], 'non_walking': []}}
    
    for class_name in classes:
        videos = all_videos[class_name]
        random.shuffle(videos)
        
        total = len(videos)
        train_count = int(total * target_ratios['train'])
        val_count = int(total * target_ratios['val'])
        test_count = total - train_count - val_count  # Remainder goes to test
        
        # Split videos
        redistribution_plan['train'][class_name] = videos[:train_count]
        redistribution_plan['val'][class_name] = videos[train_count:train_count + val_count]
        redistribution_plan['test'][class_name] = videos[train_count + val_count:]
    
    # Step 3: Show redistribution plan
    print("\n" + "="*70)
    print("REDISTRIBUTION PLAN")
    print("="*70)
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-Walking':<15} {'Total':<15} {'Ratio':<15}")
    print("-"*75)
    
    for split in splits:
        walking_count = len(redistribution_plan[split]['walking'])
        non_walking_count = len(redistribution_plan[split]['non_walking'])
        total = walking_count + non_walking_count
        ratio = (total / (len(all_videos['walking']) + len(all_videos['non_walking']))) * 100
        print(f"{split:<15} {walking_count:<15} {non_walking_count:<15} "
              f"{total:<15} {ratio:>5.1f}%")
    
    print("="*70)
    
    if dry_run:
        print("\n[DRY RUN] This will move videos to achieve 70-15-15 distribution.")
        print("Note: Files will be moved from their current locations to new splits.")
        return redistribution_plan
    
    # Step 4: Create temporary backup structure and move files
    print("\nMoving videos to achieve 70-15-15 distribution...")
    
    # Create a temporary directory structure
    temp_dir = os.path.join(dataset_dir, '_temp_redistribute')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Create new structure in temp
    for split in splits:
        for class_name in classes:
            temp_class_dir = os.path.join(temp_dir, split, class_name)
            os.makedirs(temp_class_dir, exist_ok=True)
    
    moved_count = 0
    
    # Move files to temporary structure
    for split in splits:
        for class_name in classes:
            for video_info in redistribution_plan[split][class_name]:
                source = video_info['path']
                filename = video_info['filename']
                dest = os.path.join(temp_dir, split, class_name, filename)
                
                try:
                    shutil.move(source, dest)
                    moved_count += 1
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")
    
    print(f"  Moved {moved_count} videos to temporary structure")
    
    # Step 5: Clear old directories and move from temp to final structure
    print("\nFinalizing new structure...")
    
    # First, clear old directories (they should be empty after moving to temp)
    for split in splits:
        for class_name in classes:
            old_dir = os.path.join(dataset_dir, split, class_name)
            if os.path.exists(old_dir):
                # Remove any remaining files (shouldn't happen, but just in case)
                try:
                    remaining_files = os.listdir(old_dir)
                    if remaining_files:
                        print(f"  Warning: Found {len(remaining_files)} files still in {old_dir}")
                        for filename in remaining_files:
                            old_file = os.path.join(old_dir, filename)
                            if os.path.isfile(old_file):
                                os.remove(old_file)
                    # Remove empty directory
                    os.rmdir(old_dir)
                except Exception as e:
                    print(f"  Warning: Could not remove {old_dir}: {e}")
            
            # Create new directory structure
            new_dir = os.path.join(dataset_dir, split, class_name)
            os.makedirs(new_dir, exist_ok=True)
            
            # Move files from temp to final location
            temp_class_dir = os.path.join(temp_dir, split, class_name)
            if os.path.exists(temp_class_dir):
                for filename in os.listdir(temp_class_dir):
                    source = os.path.join(temp_class_dir, filename)
                    dest = os.path.join(new_dir, filename)
                    try:
                        shutil.move(source, dest)
                    except Exception as e:
                        print(f"  Error moving {filename} to final location: {e}")
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"  Warning: Could not remove temp directory: {e}")
    
    print("✓ Redistribution complete!")
    return redistribution_plan


def balance_dataset(dataset_dir, dry_run=True):
    """Balance dataset by removing non_walking videos to match walking count"""
    splits = ['train', 'test', 'val']
    classes = ['walking', 'non_walking']
    
    counts = get_dataset_counts(dataset_dir)
    
    print("\n" + "="*70)
    print("DATASET BALANCING ANALYSIS")
    print("="*70)
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-Walking':<15} {'Remove':<15} {'Result':<15}")
    print("-"*70)
    
    to_remove = defaultdict(list)
    
    for split in splits:
        walking_count = counts[split]['walking']
        non_walking_count = counts[split]['non_walking']
        
        if non_walking_count > walking_count:
            remove_count = non_walking_count - walking_count
            target_count = walking_count
            
            # Get all non_walking videos for this split
            non_walking_dir = os.path.join(dataset_dir, split, 'non_walking')
            non_walking_videos = get_video_files(non_walking_dir)
            
            # Randomly select videos to remove
            random.seed(42)  # For reproducibility
            videos_to_remove = random.sample(non_walking_videos, remove_count)
            to_remove[split] = videos_to_remove
            
            print(f"{split:<15} {walking_count:<15} {non_walking_count:<15} "
                  f"{remove_count:<15} {target_count:<15}")
        else:
            print(f"{split:<15} {walking_count:<15} {non_walking_count:<15} "
                  f"{'0 (balanced)':<15} {walking_count:<15}")
    
    print("="*70)
    
    # Calculate totals
    total_to_remove = sum(len(videos) for videos in to_remove.values())
    
    if total_to_remove == 0:
        print("\n✓ Dataset is already balanced!")
        return 0
    
    print(f"\nTotal videos to remove: {total_to_remove}")
    
    if dry_run:
        print("\n[DRY RUN] Videos that would be removed:")
        for split in to_remove:
            print(f"\n  {split.upper()} ({len(to_remove[split])} videos):")
            for video_path in to_remove[split][:5]:  # Show first 5
                print(f"    - {os.path.basename(video_path)}")
            if len(to_remove[split]) > 5:
                print(f"    ... and {len(to_remove[split]) - 5} more")
        return total_to_remove
    else:
        print("\nRemoving videos...")
        deleted = 0
        for split in to_remove:
            print(f"\n  Removing from {split}:")
            deleted += delete_videos(to_remove[split], dry_run=False)
        print(f"\n✓ Successfully removed {deleted} videos")
        return deleted


def main():
    """Main function"""
    script_dir = Path(__file__).parent.absolute()
    dataset_dir = script_dir / "dataset"
    
    pattern = "Meet and Split_Meet and Split"
    
    print("="*70)
    print("DATASET BALANCING SCRIPT")
    print("="*70)
    print(f"\nDataset directory: {dataset_dir}")
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"\nError: Dataset directory not found at {dataset_dir}")
        return
    
    # Step 1: Find and report "Meet and Split" videos
    print("\n" + "="*70)
    print("STEP 1: FINDING 'MEET AND SPLIT' VIDEOS")
    print("="*70)
    
    total_meet_split, breakdown_meet_split, matches_meet_split = count_videos_by_pattern(
        str(dataset_dir), pattern
    )
    
    print(f"\nFound {total_meet_split} videos matching pattern: '{pattern}'")
    print("\nBreakdown by split:")
    for split in breakdown_meet_split:
        print(f"  {split}: {breakdown_meet_split[split]} videos")
        for class_name in matches_meet_split[split]:
            count = len(matches_meet_split[split][class_name])
            print(f"    - {class_name}: {count}")
    
    if total_meet_split == 0:
        print("\nNo videos found matching the pattern. Proceeding to balancing...")
    else:
        # Show sample files
        print("\nSample files to be deleted:")
        sample_shown = 0
        for split in matches_meet_split:
            for class_name in matches_meet_split[split]:
                for video_path in matches_meet_split[split][class_name][:3]:
                    if sample_shown < 5:
                        print(f"  - {video_path}")
                        sample_shown += 1
                if sample_shown >= 5:
                    break
            if sample_shown >= 5:
                break
        
        if total_meet_split > 5:
            print(f"  ... and {total_meet_split - 5} more files")
        
        # Ask for confirmation
        print(f"\n⚠️  This will delete {total_meet_split} videos.")
        response = input("Proceed with deletion? (yes/no): ").strip().lower()
        
        if response == 'yes':
            print("\nDeleting 'Meet and Split' videos...")
            deleted_count = 0
            for split in matches_meet_split:
                for class_name in matches_meet_split[split]:
                    deleted = delete_videos(matches_meet_split[split][class_name], dry_run=False)
                    deleted_count += deleted
            print(f"\n✓ Successfully deleted {deleted_count} 'Meet and Split' videos")
        else:
            print("Deletion cancelled. Exiting.")
            return
    
    # Step 2: Balance the dataset
    print("\n" + "="*70)
    print("STEP 2: BALANCING DATASET")
    print("="*70)
    
    print("\nCurrent dataset counts (after 'Meet and Split' removal):")
    current_counts = get_dataset_counts(str(dataset_dir))
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-Walking':<15}")
    print("-"*45)
    for split in ['train', 'test', 'val']:
        print(f"{split:<15} {current_counts[split]['walking']:<15} "
              f"{current_counts[split]['non_walking']:<15}")
    
    # Dry run for balancing
    print("\n[DRY RUN] Calculating balancing plan...")
    total_to_remove = balance_dataset(str(dataset_dir), dry_run=True)
    
    if total_to_remove > 0:
        print(f"\n⚠️  This will remove {total_to_remove} non_walking videos to balance the dataset.")
        response = input("Proceed with balancing? (yes/no): ").strip().lower()
        
        if response == 'yes':
            deleted = balance_dataset(str(dataset_dir), dry_run=False)
            print(f"\n✓ Dataset balancing complete! Removed {deleted} videos.")
        else:
            print("Balancing cancelled.")
    else:
        print("\n✓ Dataset is already balanced!")
    
    # Step 3: Redistribute to 70-15-15 ratio
    print("\n" + "="*70)
    print("STEP 3: REDISTRIBUTE TO 70-15-15 RATIO")
    print("="*70)
    
    print("\nCurrent distribution:")
    current_counts = get_dataset_counts(str(dataset_dir))
    total_all = sum(current_counts[split]['walking'] + current_counts[split]['non_walking'] 
                    for split in ['train', 'test', 'val'])
    
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-Walking':<15} {'Total':<15} {'Ratio':<15}")
    print("-"*75)
    for split in ['train', 'test', 'val']:
        walking = current_counts[split]['walking']
        non_walking = current_counts[split]['non_walking']
        total = walking + non_walking
        ratio = (total / total_all * 100) if total_all > 0 else 0
        print(f"{split:<15} {walking:<15} {non_walking:<15} {total:<15} {ratio:>5.1f}%")
    
    # Dry run for redistribution
    print("\n[DRY RUN] Calculating 70-15-15 redistribution plan...")
    redistribution_plan = redistribute_dataset_701515(str(dataset_dir), dry_run=True)
    
    print(f"\n⚠️  This will redistribute all videos to achieve 70-15-15 split ratio.")
    response = input("Proceed with redistribution? (yes/no): ").strip().lower()
    
    if response == 'yes':
        redistribute_dataset_701515(str(dataset_dir), dry_run=False)
        print("\n✓ Dataset redistribution complete!")
    else:
        print("Redistribution cancelled.")
    
    # Final statistics
    print("\n" + "="*70)
    print("FINAL DATASET STATISTICS")
    print("="*70)
    final_counts = get_dataset_counts(str(dataset_dir))
    final_total = sum(final_counts[split]['walking'] + final_counts[split]['non_walking'] 
                      for split in ['train', 'test', 'val'])
    
    print(f"\n{'Split':<15} {'Walking':<15} {'Non-Walking':<15} {'Total':<15} {'Ratio':<15}")
    print("-"*75)
    for split in ['train', 'test', 'val']:
        walking = final_counts[split]['walking']
        non_walking = final_counts[split]['non_walking']
        total = walking + non_walking
        ratio = (total / final_total * 100) if final_total > 0 else 0
        print(f"{split:<15} {walking:<15} {non_walking:<15} {total:<15} {ratio:>5.1f}%")
    
    print("="*70)


if __name__ == "__main__":
    main()

