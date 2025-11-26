"""
Dataset Manager - Track processed videos and manage dataset files
"""
import json
import os
import glob


class DatasetManager:
    def __init__(self, config_file='dataset_config.json', output_dir='dataset'):
        self.config_file = config_file
        self.output_dir = output_dir
        self.processed_videos = set()
        self.load_config()
    
    def load_config(self):
        """Load processed videos list from config file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.processed_videos = set(data.get('processed_videos', []))
            except Exception as e:
                print(f"Error loading config: {e}")
                self.processed_videos = set()
        else:
            self.processed_videos = set()
    
    def save_config(self):
        """Save processed videos list to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    'processed_videos': list(self.processed_videos)
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def is_processed(self, video_name):
        """Check if video has been processed before"""
        return video_name in self.processed_videos
    
    def mark_as_processed(self, video_name):
        """Mark video as processed"""
        self.processed_videos.add(video_name)
        self.save_config()
    
    def get_processed_list(self):
        """Get list of processed videos"""
        return sorted(list(self.processed_videos))
    
    def get_processed_count(self):
        """Get count of processed videos"""
        return len(self.processed_videos)
    
    def remove_dataset(self, video_name):
        """Remove all files associated with a video from dataset"""
        if not video_name:
            return 0
        
        # Remove from processed list
        if video_name in self.processed_videos:
            self.processed_videos.remove(video_name)
            self.save_config()
        
        # Get basename without extension
        basename = os.path.splitext(video_name)[0]
        
        # Find and delete all files starting with basename
        pattern = os.path.join(self.output_dir, f"{basename}_person*.mp4")
        mp4_files = glob.glob(pattern)
        
        pattern = os.path.join(self.output_dir, f"{basename}_person*.json")
        json_files = glob.glob(pattern)
        
        deleted_count = 0
        
        # Delete mp4 files
        for file in mp4_files:
            try:
                os.remove(file)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        
        # Delete json files
        for file in json_files:
            try:
                os.remove(file)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        
        return deleted_count
    
    def clear_all(self):
        """Clear all processed videos tracking"""
        self.processed_videos.clear()
        self.save_config()
    
    def get_dataset_stats(self):
        """Get statistics about dataset"""
        if not os.path.exists(self.output_dir):
            return {'videos': 0, 'mp4_files': 0, 'json_files': 0}
        
        mp4_files = glob.glob(os.path.join(self.output_dir, "*.mp4"))
        json_files = glob.glob(os.path.join(self.output_dir, "*.json"))
        
        return {
            'videos': self.get_processed_count(),
            'mp4_files': len(mp4_files),
            'json_files': len(json_files)
        }
