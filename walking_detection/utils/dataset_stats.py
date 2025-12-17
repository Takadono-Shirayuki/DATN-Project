"""
Dataset Statistics Script
Returns detailed statistics about the processed dataset including file counts and percentages
"""
import os
from pathlib import Path
from collections import defaultdict


def count_files(directory, extensions=None):
    """Count files in a directory recursively, optionally filtered by extension"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'}
    
    count = 0
    if not os.path.exists(directory):
        return count
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                count += 1
    
    return count


def get_dataset_stats(dataset_dir):
    """Get comprehensive statistics about the dataset"""
    stats = {
        'splits': {},
        'classes': defaultdict(int),
        'total': 0
    }
    
    splits = ['train', 'test', 'val']
    classes = ['walking', 'non_walking']
    
    # Count files for each split and class
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        stats['splits'][split] = {}
        split_total = 0
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            count = count_files(class_path)
            stats['splits'][split][class_name] = count
            stats['classes'][class_name] += count
            split_total += count
        
        stats['splits'][split]['total'] = split_total
        stats['total'] += split_total
    
    return stats


def format_number(num):
    """Format number with thousand separators"""
    return f"{num:,}"


def print_statistics(stats):
    """Print formatted statistics"""
    splits = ['train', 'test', 'val']
    classes = ['walking', 'non_walking']
    
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print()
    
    # Detailed breakdown by split
    print("DETAILED BREAKDOWN BY SPLIT:")
    print("-"*70)
    
    for split in splits:
        if split not in stats['splits']:
            continue
        
        split_data = stats['splits'][split]
        split_total = split_data.get('total', 0)
        
        print(f"\n{split.upper()}:")
        print(f"  {'Class':<20} {'Count':<15} {'Percentage':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        
        for class_name in classes:
            count = split_data.get(class_name, 0)
            percentage = (count / split_total * 100) if split_total > 0 else 0
            print(f"  {class_name:<20} {format_number(count):<15} {percentage:>6.2f}%")
        
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        print(f"  {'TOTAL':<20} {format_number(split_total):<15} {100.0:>6.2f}%")
    
    print()
    print("="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    # Overall totals
    total = stats['total']
    print(f"\nTotal files in dataset: {format_number(total)}")
    print()
    
    # Overall class distribution
    print("Class Distribution:")
    print(f"  {'Class':<20} {'Count':<15} {'Percentage':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    
    for class_name in classes:
        count = stats['classes'][class_name]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {class_name:<20} {format_number(count):<15} {percentage:>6.2f}%")
    
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'TOTAL':<20} {format_number(total):<15} {100.0:>6.2f}%")
    
    # Split distribution
    print()
    print("Split Distribution:")
    print(f"  {'Split':<20} {'Count':<15} {'Percentage':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    
    for split in splits:
        if split in stats['splits']:
            count = stats['splits'][split].get('total', 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {split:<20} {format_number(count):<15} {percentage:>6.2f}%")
    
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'TOTAL':<20} {format_number(total):<15} {100.0:>6.2f}%")
    
    # Class balance check
    print()
    print("="*70)
    print("CLASS BALANCE ANALYSIS")
    print("="*70)
    
    walking_total = stats['classes']['walking']
    non_walking_total = stats['classes']['non_walking']
    
    if walking_total > 0 and non_walking_total > 0:
        ratio = walking_total / non_walking_total
        print(f"\nWalking to Non-Walking Ratio: {ratio:.2f}:1")
        
        if ratio > 1.1:
            print("⚠️  Warning: Walking class has significantly more samples than Non-Walking")
        elif ratio < 0.9:
            print("⚠️  Warning: Non-Walking class has significantly more samples than Walking")
        else:
            print("✓ Classes are relatively balanced")
    
    # Split-specific balance
    print("\nClass Balance by Split:")
    print(f"  {'Split':<15} {'Walking':<15} {'Non-Walking':<15} {'Ratio':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for split in splits:
        if split in stats['splits']:
            split_data = stats['splits'][split]
            walking_count = split_data.get('walking', 0)
            non_walking_count = split_data.get('non_walking', 0)
            
            if non_walking_count > 0:
                ratio = walking_count / non_walking_count
                print(f"  {split:<15} {format_number(walking_count):<15} "
                      f"{format_number(non_walking_count):<15} {ratio:>6.2f}:1")
            else:
                print(f"  {split:<15} {format_number(walking_count):<15} "
                      f"{format_number(non_walking_count):<15} {'N/A':<15}")
    
    print("="*70)


def main():
    """Main function"""
    script_dir = Path(__file__).parent.absolute()
    dataset_dir = script_dir / "dataset"
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return
    
    # Get statistics
    stats = get_dataset_stats(str(dataset_dir))
    
    # Print statistics
    print_statistics(stats)


if __name__ == "__main__":
    main()

