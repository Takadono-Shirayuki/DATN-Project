# GR2-Project Restructuring Summary

## Changes Implemented

### 1. Dataset Tab Enhancement
- **Added custom dataset path field**: Users can now specify a custom output directory for dataset processing in the Dataset tab
- **Default behavior**: If the path field is left empty, the system defaults to `person_videos` folder
- **Config file relocation**: The `dataset_config.json` is now stored in the same directory as the dataset output for better organization

### 2. Project Structure Reorganization

#### New Folder Structure
```
GR2-Project/
├── camera_tab/
│   ├── __init__.py
│   ├── camera_module.py
│   └── camera_lib.py
├── object_box_tab/
│   ├── __init__.py
│   ├── object_box_module.py
│   ├── object_box_lib.py
│   └── object_box.py
├── recorder_tab/
│   ├── __init__.py
│   ├── recorder.py
│   └── view_person_videos.py
├── dataset_tab/
│   ├── __init__.py
│   ├── dataset_manager.py
│   └── cleanup_short_videos.py
├── utils/
│   ├── __init__.py
│   ├── detector.py
│   └── pose_visualizer.py
├── camera_ui.py (main entry point)
└── ... (other project files)
```

#### File Organization by Tab
- **camera_tab/**: Camera mode functionality
  - `camera_module.py`: Main camera processing module
  - `camera_lib.py`: Camera utility functions (segmentation)
  
- **object_box_tab/**: Object detection and tracking
  - `object_box_module.py`: Object box processing module
  - `object_box_lib.py`: Object separation and tracking utilities
  - `object_box.py`: Standalone object box script
  
- **recorder_tab/**: Video recording functionality
  - `recorder.py`: Video recording class
  - `view_person_videos.py`: Video visualization tool
  
- **dataset_tab/**: Dataset processing and management
  - `dataset_manager.py`: Dataset configuration and tracking
  - `cleanup_short_videos.py`: Dataset cleanup utility
  
- **utils/**: Shared utilities
  - `detector.py`: YOLO person detector
  - `pose_visualizer.py`: Pose visualization functions
  - `__init__.py`: Exports all utility functions for easy importing

### 3. Import System Updates
All import statements have been updated to reflect the new folder structure:
- Files now import from their respective modules (e.g., `from camera_tab.camera_module import CameraModule`)
- Each folder has an `__init__.py` file to make it a proper Python package
- The `utils` package exports commonly used functions for convenient importing

### 4. Key Features
- **Modular organization**: Files are now organized by functionality/tab, making the codebase easier to navigate
- **Clean imports**: The utils module provides a centralized location for shared utilities
- **Scalability**: New features can be easily added to their respective tab folders
- **Maintainability**: Related files are grouped together, reducing cognitive load

### Usage Notes
- The main entry point remains `camera_ui.py` in the root directory
- All imports use the new folder structure
- The dataset path can be customized in the Dataset tab UI
- Config files are now stored alongside their respective data

## Testing
The application has been tested and runs successfully with the new structure.
