"""
Simple Camera UI using tkinter
"""
import tkinter as tk
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from camera_tab.camera_module import CameraModule
from object_box_tab.object_box_module import ObjectBoxModule
from dataset_tab.dataset_manager import DatasetManager
from open_set.gei import process_video_to_gei
import os
import json
import shutil

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Form1")
        self.root.geometry("1400x750")
        
        # Variables
        self.webcam_ip = tk.StringVar()
        self.dataset_path = tk.StringVar()  # Custom dataset path
        self.gait_selected_files = []
        self.gait_processing = False
        self.loaded = False
        self.running = False
        self.current_tab = 0  # 0: Camera, 1: Object Box, 2: BR dataset, 3: Person Videos, 4: Gait Preprocess, 5: Gait Labeling
        self.update_thread = None
        self.labeling_video_playing = False  # Flag to control video playback
        self.current_labeling_video = None  # Track current video file
        self.selected_labeling_file = None  # Track selected file for labeling
        
        # Gait labels storage
        self.gait_labels_file = 'gait/gait_labels.json'
        self.gait_labels = {}  # Will be loaded later after UI setup
        self.generated_file = 'gait/Generated.json'
        self.generated_data = {}  # Track generated GEIs: {filename: label}
        
        # Dataset processing
        self.selected_files = []
        self.dataset_processing = False
        
        # FPS tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = None
        self.current_resolution = "N/A"
        
        # Modules
        self.camera_module = CameraModule()
        self.objectbox_module = ObjectBoxModule()
        self.dataset_manager = DatasetManager()
        
        # Top frame for tabs and controls
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Tab buttons
        tab_button_frame = tk.Frame(top_frame)
        tab_button_frame.pack(side=tk.LEFT, padx=5)
        
        self.camera_tab_btn = tk.Button(tab_button_frame, text="Camera", width=15, 
                                         font=("Segoe UI", 12, "bold"),
                                         command=lambda: self.switch_tab(0))
        self.camera_tab_btn.pack(side=tk.LEFT, padx=2)
        
        self.objectbox_tab_btn = tk.Button(tab_button_frame, text="Object Box", width=15,
                                            font=("Segoe UI", 12),
                                            command=lambda: self.switch_tab(1))
        self.objectbox_tab_btn.pack(side=tk.LEFT, padx=2)

        self.dataset_tab_btn = tk.Button(tab_button_frame, text="BR dataset", width=15,
                         font=("Segoe UI", 12),
                         command=lambda: self.switch_tab(2))
        self.dataset_tab_btn.pack(side=tk.LEFT, padx=2)

        self.person_videos_tab_btn = tk.Button(tab_button_frame, text="Person Videos", width=15,
                 font=("Segoe UI", 12),
                 command=lambda: self.switch_tab(3))
        self.person_videos_tab_btn.pack(side=tk.LEFT, padx=2)

        self.gait_tab_btn = tk.Button(tab_button_frame, text="Gait Preprocess", width=15,
                   font=("Segoe UI", 12),
                   command=lambda: self.switch_tab(4))
        self.gait_tab_btn.pack(side=tk.LEFT, padx=2)
        
        self.gait_labeling_tab_btn = tk.Button(tab_button_frame, text="Gait Labeling", width=15,
                  font=("Segoe UI", 12),
                  command=lambda: self.switch_tab(5))
        self.gait_labeling_tab_btn.pack(side=tk.LEFT, padx=2)
        
        # Webcam input (for Camera and Object Box tabs)
        self.webcam_frame = tk.Frame(top_frame)
        self.webcam_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(self.webcam_frame, text="Webcam IP:", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=5)
        webcam_entry = tk.Entry(self.webcam_frame, textvariable=self.webcam_ip, width=40, font=("Segoe UI", 11))
        webcam_entry.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = tk.Button(self.webcam_frame, text="Start", command=self.start_camera, 
                                    width=8, font=("Segoe UI", 11))
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(self.webcam_frame, text="Stop", command=self.stop_camera, 
                                   width=8, font=("Segoe UI", 11), state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Dataset output path (for Dataset tab)
        self.dataset_path_frame = tk.Frame(top_frame)
        
        tk.Label(self.dataset_path_frame, text="Output Path:", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=5)
        dataset_path_entry = tk.Entry(self.dataset_path_frame, textvariable=self.dataset_path, width=40, font=("Segoe UI", 11))
        dataset_path_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(self.dataset_path_frame, text="(default: person_videos)", 
                font=("Segoe UI", 9), fg="gray").pack(side=tk.LEFT, padx=2)
        
        # Main container
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create container for left panels to maintain position
        self.left_container = tk.Frame(main_frame, width=250)
        self.left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_container.pack_propagate(False)
        
        # Left panel - Camera tab content
        self.left_panel_camera = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_camera.pack(fill=tk.BOTH, expand=True)
        self.left_panel_camera.pack_propagate(False)
        
        # Camera mode label
        tk.Label(self.left_panel_camera, text="Camera  Hộp đối tượng", 
                font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # Checkboxes with larger font
        self.bbox_var = tk.BooleanVar()
        self.pose_var = tk.BooleanVar()
        self.seg_var = tk.BooleanVar()
        self.action_var = tk.BooleanVar(value=True)  # Default enabled
        
        tk.Checkbutton(self.left_panel_camera, text="Hộp giới hạn", variable=self.bbox_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_camera, text="Khung xương", variable=self.pose_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_camera, text="Phân đoạn ảnh", variable=self.seg_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_camera, text="✨ Nhận diện hành động", variable=self.action_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change, fg="#0066cc").pack(anchor=tk.W, padx=15, pady=5)
        
        # Spacer
        tk.Frame(self.left_panel_camera, height=20).pack()
        
        # Camera Info section
        info_frame = tk.Frame(self.left_panel_camera)
        info_frame.pack(side=tk.BOTTOM, pady=15, fill=tk.X, padx=10)
        
        tk.Label(info_frame, text="Camera Info", 
                font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        
        self.camera_info_label = tk.Label(info_frame, text="Resolution: N/A\nFPS: 0", 
                                         font=("Segoe UI", 11), justify=tk.LEFT, anchor=tk.W)
        self.camera_info_label.pack(anchor=tk.W, pady=(5,0))
        # Action model status
        self.action_model_label = tk.Label(info_frame, text="Action model: unknown", 
                           font=("Segoe UI", 9), fg="#888888")
        self.action_model_label.pack(anchor=tk.W, pady=(2,0))
        
        # Left panel - Object Box tab content (initially hidden)
        self.left_panel_objectbox = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_objectbox.pack_propagate(False)
        
        # Object Box mode label
        tk.Label(self.left_panel_objectbox, text="Object Box  Hộp đối tượng", 
                font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # Checkboxes for Object Box (no bbox checkbox)
        self.obj_pose_var = tk.BooleanVar()
        self.obj_seg_var = tk.BooleanVar()
        
        tk.Checkbutton(self.left_panel_objectbox, text="Khung xương", variable=self.obj_pose_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_objectbox, text="Phân đoạn ảnh", variable=self.obj_seg_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        
        # Spacer
        tk.Frame(self.left_panel_objectbox, height=20).pack()
        
        # Object Box Info label
        self.objectbox_info_label = tk.Label(self.left_panel_objectbox, text="Object Box Info", 
                font=("Segoe UI", 12, "bold"))
        self.objectbox_info_label.pack(side=tk.BOTTOM, pady=15, anchor=tk.W, padx=10)
        
        # Navigation buttons frame at bottom (moved from Camera to Object Box)
        nav_frame = tk.Frame(self.left_panel_objectbox)
        nav_frame.pack(side=tk.BOTTOM, pady=15)
        
        tk.Button(nav_frame, text="<<", width=8, font=("Segoe UI", 10), 
                 command=lambda: self.navigate_person('prev')).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text=">>", width=8, font=("Segoe UI", 10),
                 command=lambda: self.navigate_person('next')).pack(side=tk.LEFT, padx=5)
        
        # Left panel - Dataset tab content (initially hidden)
        self.left_panel_dataset_generator = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_dataset_generator.pack_propagate(False)
        
        # Dataset controls (BR dataset)
        tk.Label(self.left_panel_dataset_generator, text="Dataset Generator", 
            font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # Note: Pose and Segmentation are always enabled for dataset generation
        tk.Label(self.left_panel_dataset_generator, text="✓ Pose keypoints enabled\n✓ Segmentation mask enabled", 
            font=("Segoe UI", 9), fg="green", justify=tk.LEFT).pack(anchor=tk.W, padx=15, pady=5)
        
        # File selection
        tk.Label(self.left_panel_dataset_generator, text="Video files:", 
            font=("Segoe UI", 10, "bold")).pack(pady=(20,5), anchor=tk.W, padx=10)
        
        # File listbox
        file_frame = tk.Frame(self.left_panel_dataset_generator)
        file_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(file_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(file_frame, height=8, font=("Segoe UI", 9),
                           yscrollcommand=scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Buttons
        btn_frame = tk.Frame(self.left_panel_dataset_generator)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Add Files", width=12, 
             font=("Segoe UI", 10), command=self.add_dataset_files).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Clear", width=12, 
             font=("Segoe UI", 10), command=self.clear_dataset_files).pack(side=tk.LEFT, padx=2)
        
        # Process button
        self.process_dataset_btn = tk.Button(self.left_panel_dataset_generator, text="Process Dataset", 
                            width=20, font=("Segoe UI", 10, "bold"),
                            command=self.process_dataset)
        self.process_dataset_btn.pack(pady=15)

        # Left panel - Gait Preprocess tab content (initially hidden)
        self.left_panel_gait = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_gait.pack_propagate(False)

        tk.Label(self.left_panel_gait, text="Gait Preprocess", 
            font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)

        # Walking filter is always enabled (no checkbox required)

        # File selection for gait preprocessing
        tk.Label(self.left_panel_gait, text="Video files:", font=("Segoe UI", 10, "bold")).pack(pady=(10,5), anchor=tk.W, padx=10)
        gait_file_frame = tk.Frame(self.left_panel_gait)
        gait_file_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        gait_scrollbar = tk.Scrollbar(gait_file_frame)
        gait_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.gait_file_listbox = tk.Listbox(gait_file_frame, height=8, font=("Segoe UI", 9),
                           yscrollcommand=gait_scrollbar.set)
        self.gait_file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gait_scrollbar.config(command=self.gait_file_listbox.yview)

        gait_btn_frame = tk.Frame(self.left_panel_gait)
        gait_btn_frame.pack(pady=8)
        tk.Button(gait_btn_frame, text="Add Files", width=12, font=("Segoe UI", 10),
             command=self.add_gait_files).pack(side=tk.LEFT, padx=2)
        tk.Button(gait_btn_frame, text="Clear", width=12, font=("Segoe UI", 10),
             command=self.clear_gait_files).pack(side=tk.LEFT, padx=2)

        self.process_gait_btn = tk.Button(self.left_panel_gait, text="Process Gait", 
                          width=20, font=("Segoe UI", 10, "bold"),
                          command=self.process_gait)
        self.process_gait_btn.pack(pady=10)

        # Processed list for gait outputs
        tk.Label(self.left_panel_gait, text="Processed outputs:", font=("Segoe UI", 10, "bold")).pack(pady=(10,5), anchor=tk.W, padx=10)
        
        gait_output_frame = tk.Frame(self.left_panel_gait)
        gait_output_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        gait_output_scrollbar = tk.Scrollbar(gait_output_frame)
        gait_output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.gait_processed_listbox = tk.Listbox(gait_output_frame, height=6, font=("Segoe UI", 8),
                                                  yscrollcommand=gait_output_scrollbar.set)
        self.gait_processed_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gait_output_scrollbar.config(command=self.gait_processed_listbox.yview)
        
        # Management buttons for gait outputs
        gait_mgmt_btn_frame = tk.Frame(self.left_panel_gait)
        gait_mgmt_btn_frame.pack(pady=5)
        
        tk.Button(gait_mgmt_btn_frame, text="Delete Selected", width=12, 
                 font=("Segoe UI", 9), command=self.delete_selected_gait).pack(side=tk.LEFT, padx=2)
        tk.Button(gait_mgmt_btn_frame, text="Delete All", width=12, 
                 font=("Segoe UI", 9), command=self.delete_all_gait).pack(side=tk.LEFT, padx=2)
        tk.Button(gait_mgmt_btn_frame, text="Refresh", width=12, 
                 font=("Segoe UI", 9), command=self._refresh_gait_processed).pack(side=tk.LEFT, padx=2)
        
        # Stats label
        self.gait_stats_label = tk.Label(self.left_panel_gait, text="", 
                                         font=("Segoe UI", 8), fg="gray")
        self.gait_stats_label.pack(pady=5)
        
        # Left panel - Person Videos tab
        self.left_panel_person_videos = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_person_videos.pack_propagate(False)
        
        # Person Video Manager controls
        tk.Label(self.left_panel_person_videos, text="Person Video Manager", 
            font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # Processed videos listbox
        mgmt_frame = tk.Frame(self.left_panel_person_videos)
        mgmt_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        mgmt_scrollbar = tk.Scrollbar(mgmt_frame)
        mgmt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.processed_listbox = tk.Listbox(mgmt_frame, height=6, font=("Segoe UI", 8),
                           selectmode=tk.MULTIPLE,
                           yscrollcommand=mgmt_scrollbar.set)
        self.processed_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mgmt_scrollbar.config(command=self.processed_listbox.yview)
        
        # Management buttons
        mgmt_btn_frame = tk.Frame(self.left_panel_person_videos)
        mgmt_btn_frame.pack(pady=5)
        
        tk.Button(mgmt_btn_frame, text="Delete Selected", width=12, 
             font=("Segoe UI", 9), command=self.delete_selected_dataset).pack(side=tk.LEFT, padx=2)
        tk.Button(mgmt_btn_frame, text="Delete All", width=12, 
             font=("Segoe UI", 9), command=self.delete_all_datasets).pack(side=tk.LEFT, padx=2)
        tk.Button(mgmt_btn_frame, text="Refresh", width=12, 
             font=("Segoe UI", 9), command=self.refresh_dataset_list).pack(side=tk.LEFT, padx=2)
        
        # Stats label
        self.dataset_stats_label = tk.Label(self.left_panel_person_videos, text="", 
                           font=("Segoe UI", 8), fg="gray")
        self.dataset_stats_label.pack(pady=5)
        
        # Left panel - Gait Labeling tab
        self.left_panel_labeling = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_labeling.pack_propagate(False)
        
        tk.Label(self.left_panel_labeling, text="Gait Labeling & GEI", 
                font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # File selection from preprocessed
        tk.Label(self.left_panel_labeling, text="Preprocessed files:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(10,5), anchor=tk.W, padx=10)
        
        labeling_file_frame = tk.Frame(self.left_panel_labeling)
        labeling_file_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        labeling_scrollbar = tk.Scrollbar(labeling_file_frame)
        labeling_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.labeling_file_listbox = tk.Listbox(labeling_file_frame, height=8, font=("Segoe UI", 9),
                                                yscrollcommand=labeling_scrollbar.set)
        self.labeling_file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        labeling_scrollbar.config(command=self.labeling_file_listbox.yview)
        self.labeling_file_listbox.bind('<<ListboxSelect>>', self.on_labeling_file_selected)
        
        tk.Button(self.left_panel_labeling, text="Refresh List", width=20, 
                 font=("Segoe UI", 9), command=self.refresh_labeling_list).pack(pady=5)
        
        # Label assignment
        tk.Label(self.left_panel_labeling, text="Assign Label:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(15,5), anchor=tk.W, padx=10)
        
        # Current label status
        self.current_label_status = tk.Label(self.left_panel_labeling, text="", 
                                            font=("Segoe UI", 8), fg="gray", wraplength=280)
        self.current_label_status.pack(pady=(0,5), padx=10, anchor=tk.W)
        
        label_frame = tk.Frame(self.left_panel_labeling)
        label_frame.pack(padx=10, pady=5, fill=tk.X)
        
        tk.Label(label_frame, text="Person ID:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0,5))
        self.label_entry = tk.Entry(label_frame, width=15, font=("Segoe UI", 10))
        self.label_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Button(self.left_panel_labeling, text="Save Label", width=20, 
                 font=("Segoe UI", 10), command=self.save_label).pack(pady=10)
        
        # GEI generation
        tk.Label(self.left_panel_labeling, text="GEI Generation:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(15,5), anchor=tk.W, padx=10)
        
        self.current_file_label = tk.Label(self.left_panel_labeling, text="", 
                                          font=("Segoe UI", 8), fg="blue")
        self.current_file_label.pack(pady=5)
        
        tk.Button(self.left_panel_labeling, text="Generate", width=20, 
                 font=("Segoe UI", 10, "bold"), fg="#0066cc", command=self.generate_gei_all).pack(pady=5)
        tk.Button(self.left_panel_labeling, text="Regenerate All", width=20, 
                 font=("Segoe UI", 10, "bold"), command=self.regenerate_all_gei).pack(pady=5)
        
        # Status for labeling
        self.labeling_status_label = tk.Label(self.left_panel_labeling, text="", 
                                             font=("Segoe UI", 8), fg="gray")
        self.labeling_status_label.pack(pady=10)
        
        # Right panel (video display)
        right_panel = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2, bg="black")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video label (main display)
        self.video_label = tk.Label(right_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Info overlay frame (for dataset progress, etc)
        self.info_overlay_frame = tk.Frame(right_panel, bg="#1e1e1e", relief=tk.RAISED, borderwidth=2)
        
        # Dataset processing info (hidden by default)
        self.dataset_info_container = tk.Frame(self.info_overlay_frame, bg="#1e1e1e")
        self.dataset_info_container.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(self.dataset_info_container, text="Dataset Processing", 
                font=("Segoe UI", 14, "bold"), bg="#1e1e1e", fg="white").pack(pady=(0,15))
        
        # Progress bar style info
        self.dataset_current_file_label = tk.Label(self.dataset_info_container, 
                                                   text="", 
                                                   font=("Segoe UI", 11), 
                                                   bg="#1e1e1e", fg="#00ff00",
                                                   wraplength=600)
        self.dataset_current_file_label.pack(pady=5)
        
        self.dataset_frame_progress_label = tk.Label(self.dataset_info_container, 
                                                     text="", 
                                                     font=("Segoe UI", 10), 
                                                     bg="#1e1e1e", fg="#aaaaaa")
        self.dataset_frame_progress_label.pack(pady=5)
        
        self.dataset_summary_label = tk.Label(self.dataset_info_container, 
                                              text="", 
                                              font=("Segoe UI", 11, "bold"), 
                                              bg="#1e1e1e", fg="#ffaa00")
        self.dataset_summary_label.pack(pady=15)
        
        # Status label
        self.status_label = tk.Label(root, text="Ready", anchor=tk.W, font=("Segoe UI", 10))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Initialize tab state
        self.switch_tab(0)
    
    def switch_tab(self, tab_index):
        """Switch between Camera, Object Box, and Recorder tabs"""
        self.current_tab = tab_index
        
        # Reset all tab buttons
        self.camera_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        self.objectbox_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        self.dataset_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        self.person_videos_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        self.gait_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        self.gait_labeling_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        
        # Hide all panels
        self.left_panel_camera.pack_forget()
        self.left_panel_objectbox.pack_forget()
        self.left_panel_dataset_generator.pack_forget()
        self.left_panel_person_videos.pack_forget()
        self.left_panel_gait.pack_forget()
        self.left_panel_labeling.pack_forget()
        
        if tab_index == 0:  # Camera tab
            self.camera_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_camera.pack(fill=tk.BOTH, expand=True)
            
            # Show webcam controls, hide dataset path
            self.webcam_frame.pack(side=tk.LEFT, padx=20)
            self.dataset_path_frame.pack_forget()
            
            self.status_label.config(text="Camera tab selected")
            
        elif tab_index == 1:  # Object Box tab
            self.objectbox_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_objectbox.pack(fill=tk.BOTH, expand=True)
            
            # Show webcam controls, hide dataset path
            self.webcam_frame.pack(side=tk.LEFT, padx=20)
            self.dataset_path_frame.pack_forget()
            
            self.status_label.config(text="Object Box tab selected")
            
        elif tab_index == 2:  # BR dataset tab (Dataset Generator)
            self.dataset_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_dataset_generator.pack(fill=tk.BOTH, expand=True)
            
            # Hide webcam controls, show dataset path
            self.webcam_frame.pack_forget()
            self.dataset_path_frame.pack(side=tk.LEFT, padx=20)

            # Show info overlay if processing
            if self.dataset_processing:
                self.info_overlay_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=700, height=300)

            self.status_label.config(text="BR dataset tab selected")

        elif tab_index == 3:  # Person Videos tab
            self.person_videos_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_person_videos.pack(fill=tk.BOTH, expand=True)
            
            # Hide webcam controls, show dataset path
            self.webcam_frame.pack_forget()
            self.dataset_path_frame.pack(side=tk.LEFT, padx=20)
            
            # Show info overlay if processing
            if self.dataset_processing:
                self.info_overlay_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=700, height=300)
            
            # Refresh dataset list
            self.refresh_dataset_list()
            
            self.status_label.config(text="Person Videos tab selected")
            
        elif tab_index == 4:  # Gait preprocess tab
            self.gait_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_gait.pack(fill=tk.BOTH, expand=True)

            # Hide webcam controls, show dataset path (optional)
            self.webcam_frame.pack_forget()
            self.dataset_path_frame.pack_forget()

            # Refresh processed outputs list
            self._refresh_gait_processed()

            self.status_label.config(text="Gait Preprocess tab selected")

        else:  # Gait Labeling tab (tab_index == 5)
            self.gait_labeling_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_labeling.pack(fill=tk.BOTH, expand=True)
            
            # Hide both controls
            self.webcam_frame.pack_forget()
            self.dataset_path_frame.pack_forget()
            
            # Load labels if not loaded yet
            if not self.gait_labels:
                self.gait_labels = self._load_gait_labels()
            
            # Refresh file list
            self.refresh_labeling_list()
            
            self.status_label.config(text="Gait Labeling tab selected")
    
    def load_camera(self):
        """Load camera/video source and initialize modules"""
        if self.loaded:
            self.status_label.config(text="Already loaded")
            return
        
        input_source = self.webcam_ip.get() or "0"
        
        if self.current_tab == 0:  # Camera mode
            success = self.camera_module.start(input_source)
            if not success:
                self.status_label.config(text="Failed to load camera")
                return
            # Stop immediately after load
            self.camera_module.stop()
        elif self.current_tab == 1:  # Object Box mode
            success = self.objectbox_module.start(input_source)
            if not success:
                self.status_label.config(text="Failed to load object box")
                return
            # Stop immediately after load
            self.objectbox_module.stop()
        else:
            self.status_label.config(text="Recorder mode not implemented yet")
            return
        
        self.loaded = True
        self.status_label.config(text="Loaded - Ready to start")
    
    def start_camera(self):
        """Start processing"""
        # Auto-load if not loaded
        if not self.loaded:
            self.load_camera()
        
        if self.running:
            return
        
        input_source = self.webcam_ip.get() or "0"
        
        if self.current_tab == 0:  # Camera mode
            success = self.camera_module.start(input_source)
            if not success:
                self.status_label.config(text="Failed to start camera")
                return
            # Update module options from checkboxes
            self.camera_module.set_options(
                bbox=self.bbox_var.get(),
                pose=self.pose_var.get(),
                segmentation=self.seg_var.get(),
                action_recognition=self.action_var.get()
            )
            # Update action model status label
            try:
                if self.camera_module.action_recognizer is not None:
                    self.action_model_label.config(text="Action model: loaded", fg="#00aa00")
                else:
                    self.action_model_label.config(text="Action model: missing", fg="#aa0000")
            except Exception:
                pass
        elif self.current_tab == 1:  # Object Box mode
            success = self.objectbox_module.start(input_source)
            if not success:
                self.status_label.config(text="Failed to start object box")
                return
            # Update module options from checkboxes
            self.objectbox_module.set_options(
                pose=self.obj_pose_var.get(),
                segmentation=self.obj_seg_var.get()
            )
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Running")
        
        # Reset FPS tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = None
        
        # Start UI update thread
        self.update_thread = threading.Thread(target=self._update_display, daemon=True)
        self.update_thread.start()
    
    def stop_camera(self):
        """Stop processing"""
        if not self.running:
            return
        
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped")
        
        # Stop modules
        self.camera_module.stop()
        self.objectbox_module.stop()
    
    def _handle_video_ended(self):
        """Handle when video ends naturally (not user stopped)"""
        if not self.running:
            return
        
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Video Ended")
    
    def _update_display(self):
        """Update display from module in separate thread"""
        import time
        
        while self.running:
            try:
                # Check if video has ended
                if self.current_tab == 0:
                    if not self.camera_module.is_active():
                        self.root.after(0, self._handle_video_ended)
                        break
                elif self.current_tab == 1:
                    if not self.objectbox_module.is_active():
                        self.root.after(0, self._handle_video_ended)
                        break
                
                if self.current_tab == 0:  # Camera mode
                    frame = self.camera_module.get_frame(timeout=0.1)
                    if frame is not None:
                        self.display_frame(frame)
                        self._update_fps_info(frame)
                        
                elif self.current_tab == 1:  # Object Box mode
                    frame, metadata = self.objectbox_module.get_frame(timeout=0.1)
                    if frame is not None:
                        self.display_frame(frame)
                        self._update_fps_info(frame)
                        # Update info label
                        if metadata:
                            info_text = f"Person: {metadata.get('selected_id', 'None')}\nTotal: {metadata.get('person_count', 0)}"
                            self.objectbox_info_label.config(text=info_text)
            except Exception as e:
                print(f"Update error: {e}")
    
    def _update_fps_info(self, frame):
        """Update FPS and resolution info"""
        import time
        
        # Update resolution
        if frame is not None:
            h, w = frame.shape[:2]
            self.current_resolution = f"{w}x{h}"
        
        # Calculate FPS
        current_time = time.time()
        if self.last_fps_update is None:
            self.last_fps_update = current_time
            self.frame_count = 0
        
        self.frame_count += 1
        elapsed = current_time - self.last_fps_update
        
        # Update FPS every second
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
            
            # Update UI label
            if self.current_tab == 0:  # Camera tab
                info_text = f"Resolution: {self.current_resolution}\nFPS: {self.fps:.1f}"
                self.camera_info_label.config(text=info_text)
    
    def display_frame(self, frame):
        """Display frame in GUI"""
        try:
            if frame is None:
                return
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display area
            h, w = frame_rgb.shape[:2]
            max_h, max_w = 600, 1100
            scale = min(max_w/w, max_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label (must be done in main thread)
            self.root.after(0, self._update_image_label, imgtk)
        except Exception as e:
            print(f"Display error: {e}")
    
    def _update_image_label(self, imgtk):
        """Update image label (called from main thread)"""
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def navigate_person(self, direction):
        """Navigate between detected persons in Object Box mode"""
        if self.current_tab == 1:  # Object Box mode
            self.objectbox_module.navigate_person(direction)
    
    def on_checkbox_change(self):
        """Update module options when checkboxes change"""
        if not self.running:
            return
        
        if self.current_tab == 0:  # Camera mode
            self.camera_module.set_options(
                bbox=self.bbox_var.get(),
                pose=self.pose_var.get(),
                segmentation=self.seg_var.get(),
                action_recognition=self.action_var.get()
            )
        elif self.current_tab == 1:  # Object Box mode
            self.objectbox_module.set_options(
                pose=self.obj_pose_var.get(),
                segmentation=self.obj_seg_var.get()
            )
    
    def add_dataset_files(self):
        """Open file dialog to add video files"""
        from tkinter import filedialog
        
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                # Display only filename in listbox
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        self.status_label.config(text=f"Added {len(files)} file(s). Total: {len(self.selected_files)}")

    def add_gait_files(self):
        """Open file dialog to add video files for gait preprocessing"""
        from tkinter import filedialog
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        for file in files:
            if file not in self.gait_selected_files:
                self.gait_selected_files.append(file)
                self.gait_file_listbox.insert(tk.END, os.path.basename(file))

        self.status_label.config(text=f"Added {len(files)} gait file(s). Total: {len(self.gait_selected_files)}")

    def clear_gait_files(self):
        """Clear selected gait files"""
        self.gait_selected_files.clear()
        self.gait_file_listbox.delete(0, tk.END)
        self.status_label.config(text="Gait file list cleared")

    def _refresh_gait_processed(self):
        """Refresh processed outputs list for gait tab"""
        self.gait_processed_listbox.delete(0, tk.END)
        out_dir = self.dataset_path.get().strip() or 'gait/preprocess'
        if os.path.exists(out_dir):
            # list mp4 files
            files = [f for f in os.listdir(out_dir) if f.endswith('.mp4')]
            for f in sorted(files):
                self.gait_processed_listbox.insert(tk.END, f)
            
            # Update stats
            self.gait_stats_label.config(text=f"Total: {len(files)} files")

    def delete_selected_gait(self):
        """Delete selected gait output files"""
        selection = self.gait_processed_listbox.curselection()
        if not selection:
            self.status_label.config(text="No file selected")
            return
        
        out_dir = self.dataset_path.get().strip() or 'gait/preprocess'
        selected_files = [self.gait_processed_listbox.get(i) for i in selection]
        
        for filename in selected_files:
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        
        self._refresh_gait_processed()
        self.status_label.config(text=f"Deleted {len(selected_files)} file(s)")

    def delete_all_gait(self):
        """Delete all gait output files"""
        out_dir = self.dataset_path.get().strip() or 'gait/preprocess'
        if not os.path.exists(out_dir):
            return
        
        files = [f for f in os.listdir(out_dir) if f.endswith('.mp4')]
        if not files:
            self.status_label.config(text="No files to delete")
            return
        
        # Confirm dialog
        import tkinter.messagebox as messagebox
        if not messagebox.askyesno("Confirm Delete", f"Delete all {len(files)} files?"):
            return
        
        for filename in files:
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        
        self._refresh_gait_processed()
        self.status_label.config(text=f"Deleted {len(files)} file(s)")

    def process_gait(self):
        """Process gait files with optional walking filter"""
        if not self.gait_selected_files:
            self.status_label.config(text="No gait files selected!")
            return

        if self.gait_processing:
            self.status_label.config(text="Gait processing already running...")
            return

        output_dir = self.dataset_path.get().strip() or 'gait/preprocess'
        self.dataset_manager.set_output_dir(output_dir)

        self.process_gait_btn.config(state=tk.DISABLED)
        self.gait_processing = True

        # Show overlay
        self.info_overlay_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=700, height=300)

        threading.Thread(target=self._process_gait_thread, daemon=True).start()

    def _process_gait_thread(self):
        """Background thread to process gait files and filter walking segments"""
        import cv2
        from ultralytics import YOLO
        from object_box_tab.object_box_lib import separate_object_gpu_tracking

        # Prepare output dir
        output_dir = self.dataset_manager.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load models
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        self.root.after(0, lambda: self.status_label.config(text=f"Loading models on {device}..."))
        seg_model = YOLO('yolov8s-seg.pt').to(device)
        pose_model = YOLO('yolov8s-pose.pt').to(device)

        # Try to load action recognizer (attempt always). If missing, fall back to no-filter behavior.
        action_recognizer = None
        try:
            from action_classifier_module import ActionRecognizer
            model_path = os.path.join('Behavier_recognition', 'action_classifier_128_2.pth')
            if not os.path.exists(model_path):
                model_path = 'action_classifier_128_2.pth'
            if os.path.exists(model_path):
                action_recognizer = ActionRecognizer(model_path=model_path)
            else:
                action_recognizer = None
        except Exception:
            action_recognizer = None

        total_files = len(self.gait_selected_files)
        total_persons_all = 0

        for idx, video_path in enumerate(self.gait_selected_files, 1):
            try:
                filename = os.path.basename(video_path)
                basename = os.path.splitext(filename)[0]

                self.root.after(0, lambda f=idx, t=total_files, n=filename: 
                                self.dataset_current_file_label.config(text=f"Processing {f}/{t}:\n{n}"))

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_idx = 0
                # Per-person segment state: { person_id: { 'seg_idx': int, 'recorder': Recorder|None, 'base_name': str } }
                person_segments = {}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Store original frame for later use
                    original_frame = frame.copy()

                    # Use scale_factor=1.0 so bbox/keypoints are in original coords
                    output_frames = separate_object_gpu_tracking(
                        frame,
                        seg_model=seg_model,
                        pose_model=pose_model,
                        scale_factor=1.0,
                        tracker='bytetrack.yaml',
                        enable_segmentation=True
                    )

                    for person_id, data in output_frames.items():
                        # initialize per-person segment state
                        if person_id not in person_segments:
                            person_num = person_id.split('_')[-1]
                            base_name = f"{basename}_person{person_num.zfill(2)}"
                            person_segments[person_id] = {
                                'seg_idx': 0,
                                'recorder_seg': None,  # Recorder for segmented version
                                'recorder_orig': None,  # Recorder for original version
                                'base_name': base_name,
                                'last_action': None,
                                'predict_count': 0
                            }

                        seg_state = person_segments[person_id]

                        # Prefer original (frame-space) keypoints for action recognizer
                        kp = data.get('orig_keypoints') or data.get('keypoints')
                        bbox = data.get('bbox')

                        is_walking = False
                        if action_recognizer is not None and kp is not None and bbox is not None:
                            try:
                                action_recognizer.update(person_id, kp, bbox)
                                # Only predict every 30 frames (matching camera tab behavior: 1 predict per sequence_length)
                                seg_state['predict_count'] += 1
                                if seg_state['predict_count'] >= 30:
                                    seg_state['predict_count'] = 0
                                    res = action_recognizer.predict(person_id, use_smoothing=True)
                                    if res.get('ready') and res.get('action') == 'Walking':
                                        is_walking = True
                                        seg_state['last_action'] = 'Walking'
                                    else:
                                        seg_state['last_action'] = res.get('action')
                                else:
                                    # Use last known action until next predict
                                    if seg_state['last_action'] == 'Walking':
                                        is_walking = True
                            except Exception:
                                pass

                        # Segment logic: start a new segment when walking detected, stop when walking ends
                        from recorder_tab.recorder import Recorder
                        if is_walking:
                            if seg_state['recorder_seg'] is None:
                                # start new recorders for this walking segment (both segmented and original)
                                seg_name = f"{seg_state['base_name']}_seg{seg_state['seg_idx']:02d}"
                                
                                # Recorder for segmented version
                                seg_state['recorder_seg'] = Recorder(
                                    out_dir=output_dir,
                                    fps=fps,
                                    frame_size=(224, 224),
                                    timeout=2.0,
                                    min_duration=0.0,
                                    save_json=False
                                )
                                
                                # Recorder for original version (with _orig suffix)
                                seg_state['recorder_orig'] = Recorder(
                                    out_dir=output_dir,
                                    fps=fps,
                                    frame_size=(224, 224),
                                    timeout=2.0,
                                    min_duration=0.0,
                                    save_json=False
                                )
                                
                                seg_state['seg_idx'] += 1
                            
                            # write both versions to recorders
                            try:
                                seg_pid = seg_state['base_name'] + f"_seg{seg_state['seg_idx']-1:02d}"
                                
                                # Write segmented version (from separate_object_gpu_tracking)
                                seg_state['recorder_seg'].update(
                                    pid=seg_pid, 
                                    frame=data.get('image'),  # This is the segmented 224x224
                                    keypoints=None
                                )
                                
                                # Write original version: crop same region from original frame but no segmentation
                                # Use same transformation as segmented version
                                x1, y1, x2, y2 = bbox
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Calculate center and box size (same as in crop_and_resize_gpu)
                                bbox_w = x2 - x1
                                bbox_h = y2 - y1
                                max_side = max(bbox_w, bbox_h)
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # Crop square region
                                crop_x1 = int(center_x - max_side / 2)
                                crop_y1 = int(center_y - max_side / 2)
                                crop_x2 = crop_x1 + max_side
                                crop_y2 = crop_y1 + max_side
                                
                                # Handle boundaries
                                img_h, img_w = original_frame.shape[:2]
                                canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                                
                                # Calculate intersection
                                src_x1 = max(0, crop_x1)
                                src_y1 = max(0, crop_y1)
                                src_x2 = min(img_w, crop_x2)
                                src_y2 = min(img_h, crop_y2)
                                
                                dst_x1 = max(0, -crop_x1)
                                dst_y1 = max(0, -crop_y1)
                                dst_x2 = dst_x1 + (src_x2 - src_x1)
                                dst_y2 = dst_y1 + (src_y2 - src_y1)
                                
                                if src_x1 < src_x2 and src_y1 < src_y2:
                                    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = original_frame[src_y1:src_y2, src_x1:src_x2]
                                
                                # Resize to 224x224 with aspect ratio preservation (same as crop_and_resize_gpu)
                                scale = min(224.0 / max_side, 224.0 / max_side)
                                new_w = max(1, int(max_side * scale))
                                new_h = max(1, int(max_side * scale))
                                resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                
                                # Center in 224x224 canvas
                                final = np.zeros((224, 224, 3), dtype=np.uint8)
                                x_off = (224 - new_w) // 2
                                y_off = (224 - new_h) // 2
                                final[y_off:y_off + new_h, x_off:x_off + new_w] = resized
                                
                                seg_state['recorder_orig'].update(
                                    pid=seg_pid + "_orig",
                                    frame=final,
                                    keypoints=None
                                )
                            except Exception as e:
                                pass
                        else:
                            # not walking: if recorders are open, close them
                            if seg_state['recorder_seg'] is not None:
                                seg_state['recorder_seg'].close_all()
                                seg_state['recorder_seg'] = None
                            if seg_state['recorder_orig'] is not None:
                                seg_state['recorder_orig'].close_all()
                                seg_state['recorder_orig'] = None

                    frame_idx += 1

                # Close any open segment recorders and register saved segment files
                import glob
                total_segments_saved = 0
                for person_id, seg_state in person_segments.items():
                    if seg_state.get('recorder_seg') is not None:
                        seg_state['recorder_seg'].close_all()
                        seg_state['recorder_seg'] = None
                    if seg_state.get('recorder_orig') is not None:
                        seg_state['recorder_orig'].close_all()
                        seg_state['recorder_orig'] = None
                    # Find saved segment files for this person
                    pattern = os.path.join(output_dir, f"{seg_state['base_name']}_seg*.mp4")
                    saved_files = sorted(glob.glob(pattern))
                    for fpath in saved_files:
                        fname = os.path.basename(fpath)
                        self.root.after(0, lambda p=fname: self.gait_processed_listbox.insert(tk.END, p))
                        print(f"Saved clip: {fpath}")
                        total_segments_saved += 1

                cap.release()
                total_persons_all += total_segments_saved
                self.dataset_manager.mark_as_processed(filename)

            except Exception as e:
                print(f"Error processing gait {video_path}: {e}")
                import traceback
                traceback.print_exc()

        # Done
        self.gait_processing = False
        self.root.after(0, lambda: self.process_gait_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.dataset_current_file_label.config(text=f"✓ Gait processing complete!", fg="#00ff00"))
        self.root.after(0, lambda: self.dataset_frame_progress_label.config(text=f"Processed {total_files} files"))
        self.root.after(0, lambda d=output_dir: self.dataset_summary_label.config(text=f"Total persons processed: {total_persons_all}\nSaved to: {d}/", fg="#00ff00"))
        self.root.after(3000, lambda: self.info_overlay_frame.place_forget())
    
    def clear_dataset_files(self):
        """Clear all selected files"""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.status_label.config(text="File list cleared")
    
    def refresh_dataset_list(self):
        """Refresh processed videos list"""
        self.processed_listbox.delete(0, tk.END)
        
        processed = self.dataset_manager.get_processed_list()
        for video in processed:
            self.processed_listbox.insert(tk.END, video)
        
        # Update stats
        stats = self.dataset_manager.get_dataset_stats()
        stats_text = f"Processed: {stats['videos']} videos | Files: {stats['mp4_files']} MP4, {stats['json_files']} JSON"
        self.dataset_stats_label.config(text=stats_text)
        
        self.status_label.config(text=f"Refreshed: {len(processed)} processed videos")
    
    def delete_selected_dataset(self):
        """Delete selected datasets from list (supports multiple selection)"""
        selection = self.processed_listbox.curselection()
        if not selection:
            self.status_label.config(text="No dataset selected!")
            return
        
        # Get all selected video names
        video_names = [self.processed_listbox.get(idx) for idx in selection]
        
        # Confirm deletion
        from tkinter import messagebox
        confirm = messagebox.askyesno(
            "Delete Datasets",
            f"Delete {len(video_names)} dataset(s)?\n\nThis will remove all person videos and JSON files."
        )
        
        if not confirm:
            return
        
        # Delete files for each selected dataset
        total_deleted = 0
        for video_name in video_names:
            deleted_count = self.dataset_manager.remove_dataset(video_name)
            total_deleted += deleted_count
        
        # Refresh list
        self.refresh_dataset_list()
        
        self.status_label.config(text=f"Deleted {total_deleted} files from {len(video_names)} dataset(s)")
    
    def delete_all_datasets(self):
        """Delete all datasets from list"""
        processed_videos = self.dataset_manager.processed_videos
        
        if not processed_videos:
            self.status_label.config(text="No datasets to delete!")
            return
        
        # Confirm deletion
        from tkinter import messagebox
        confirm = messagebox.askyesno(
            "Delete All Datasets",
            f"Delete ALL {len(processed_videos)} datasets?\n\nThis will remove all processed videos and JSON files.\n\nThis action cannot be undone!"
        )
        
        if not confirm:
            return
        
        # Delete all datasets
        total_deleted = 0
        for video_name in processed_videos:
            deleted_count = self.dataset_manager.remove_dataset(video_name)
            total_deleted += deleted_count
        
        # Refresh list
        self.refresh_dataset_list()
        
        self.status_label.config(text=f"Deleted all datasets ({total_deleted} files removed)")
    
    def process_dataset(self):
        """Process all selected video files"""
        if not self.selected_files:
            self.status_label.config(text="No files selected!")
            return
        
        if self.dataset_processing:
            self.status_label.config(text="Already processing...")
            return
        
        # Get output directory (default to 'person_videos' if empty)
        output_dir = self.dataset_path.get().strip() or 'person_videos'
        
        # Update dataset manager with new output directory
        self.dataset_manager.set_output_dir(output_dir)
        
        # Disable button during processing
        self.process_dataset_btn.config(state=tk.DISABLED)
        self.dataset_processing = True
        
        # Show info overlay
        self.info_overlay_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=700, height=300)
        
        # Start processing in separate thread
        threading.Thread(target=self._process_dataset_thread, daemon=True).start()
    
    def _process_dataset_thread(self):
        """Process dataset in separate thread"""
        import cv2
        from ultralytics import YOLO
        from object_box_tab.object_box_lib import separate_object_gpu_tracking
        
        # Get output directory from dataset manager
        output_dir = self.dataset_manager.output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine device (CUDA > MPS > CPU)
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Metal Performance Shaders for Mac
        else:
            device = 'cpu'
        
        # Load models (always load both for dataset generation)
        self.root.after(0, lambda: self.status_label.config(text=f"Loading models on {device}..."))
        seg_model = YOLO('yolov8s-seg.pt').to(device)
        pose_model = YOLO('yolov8s-pose.pt').to(device)
        
        total_files = len(self.selected_files)
        total_persons_all = 0
        
        for file_idx, video_path in enumerate(self.selected_files, 1):
            try:
                # Update progress
                filename = os.path.basename(video_path)
                basename = os.path.splitext(filename)[0]
                
                # Check if already processed
                if self.dataset_manager.is_processed(filename):
                    print(f"⊘ Skipping {filename} (already processed)")
                    self.root.after(0, lambda f=file_idx, t=total_files, n=filename: 
                        self.dataset_current_file_label.config(text=f"Skipping {f}/{t}:\n{n} (already processed)"))
                    continue
                
                self.root.after(0, lambda f=file_idx, t=total_files, n=filename: 
                    self.dataset_current_file_label.config(text=f"Processing file {f}/{t}:\n{n}"))
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open: {video_path}")
                    continue
                
                # Get video info
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Create recorders for each person (dict: person_id -> recorder)
                from recorder_tab.recorder import Recorder
                person_recorders = {}
                
                frame_idx = 0
                persons_in_video = set()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame (always enable both pose and segmentation)
                    output_frames = separate_object_gpu_tracking(
                        frame,
                        seg_model=seg_model,
                        pose_model=pose_model,
                        scale_factor=0.5,
                        tracker='bytetrack.yaml',
                        enable_segmentation=True  # Always enabled for dataset
                    )
                    
                    # Record each person
                    for person_id, data in output_frames.items():
                        persons_in_video.add(person_id)
                        
                        # Create recorder if not exists
                        if person_id not in person_recorders:
                            # Extract person number from person_id (e.g., "Person_1" -> "01")
                            person_num = person_id.split('_')[-1]
                            output_name = f"{basename}_person{person_num.zfill(2)}"
                            
                            from recorder_tab.recorder import Recorder
                            person_recorders[person_id] = {
                                'recorder': Recorder(
                                    out_dir=output_dir,
                                    fps=fps,
                                    frame_size=(224, 224),
                                    timeout=2.0,
                                    min_duration=2.0  # Filter out videos < 2 seconds
                                ),
                                'output_name': output_name
                            }
                        
                        # Update recorder with custom naming
                        recorder_info = person_recorders[person_id]
                        recorder_info['recorder'].update(
                            pid=recorder_info['output_name'],
                            frame=data['image'],
                            keypoints=data.get('keypoints')
                        )
                    
                    frame_idx += 1
                    
                    # Update progress every 30 frames
                    if frame_idx % 30 == 0:
                        percent = int((frame_idx / total_frames) * 100)
                        progress = f"Frame {frame_idx}/{total_frames} ({percent}%)"
                        persons_found = len(persons_in_video)
                        summary = f"Persons detected: {persons_found}"
                        self.root.after(0, lambda p=progress, s=summary: (
                            self.dataset_frame_progress_label.config(text=p),
                            self.dataset_summary_label.config(text=s)
                        ))
                
                # Close all recorders for this video
                for person_id, recorder_info in person_recorders.items():
                    recorder_info['recorder'].close_all()
                
                cap.release()
                
                # Mark as processed
                self.dataset_manager.mark_as_processed(filename)
                
                total_persons_all += len(persons_in_video)
                print(f"✓ {filename}: {len(persons_in_video)} persons, {frame_idx} frames")
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Done
        self.dataset_processing = False
        self.root.after(0, lambda: self.process_dataset_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.dataset_current_file_label.config(
            text=f"✓ Processing Complete!", fg="#00ff00"))
        self.root.after(0, lambda: self.dataset_frame_progress_label.config(
            text=f"Processed {total_files} video file(s)"))
        output_dir = self.dataset_manager.output_dir
        self.root.after(0, lambda d=output_dir: self.dataset_summary_label.config(
            text=f"Total: {total_persons_all} persons extracted\nSaved to: {d}/", 
            fg="#00ff00"))
        self.root.after(0, lambda: self.status_label.config(
            text=f"Dataset processing complete: {total_persons_all} persons extracted"))
        
        # Hide overlay after 3 seconds
        self.root.after(3000, lambda: self.info_overlay_frame.place_forget())
    
    # Gait Labeling functions
    def refresh_labeling_list(self):
        """Refresh list of preprocessed files for labeling"""
        self.labeling_file_listbox.delete(0, tk.END)
        out_dir = 'gait/preprocess'
        if os.path.exists(out_dir):
            # Only show non-_orig files
            files = [f for f in os.listdir(out_dir) if f.endswith('.mp4') and '_orig' not in f]
            labeled_count = 0
            for f in sorted(files):
                # Add label info if exists
                label = self.gait_labels.get(f, "")
                if label:
                    display_text = f"{f} → [{label}]"
                    labeled_count += 1
                else:
                    display_text = f"{f} [chưa gán nhãn]"
                self.labeling_file_listbox.insert(tk.END, display_text)
            self.labeling_status_label.config(text=f"Total: {len(files)} files ({labeled_count} đã gán nhãn)")
    
    def on_labeling_file_selected(self, event):
        """Handle file selection in labeling tab"""
        selection = self.labeling_file_listbox.curselection()
        if not selection:
            return
        
        display_text = self.labeling_file_listbox.get(selection[0])
        # Extract filename from display text (remove label part)
        if ' → [' in display_text:
            filename = display_text.split(' → [')[0]
        elif ' [chưa gán nhãn]' in display_text:
            filename = display_text.replace(' [chưa gán nhãn]', '')
        else:
            filename = display_text
        
        # Lưu filename vào biến instance
        self.selected_labeling_file = filename
        self.current_file_label.config(text=f"Selected: {filename}")
        
        # Load label from JSON if available
        current_label = self.gait_labels.get(filename, "")
        self.label_entry.delete(0, tk.END)
        self.label_entry.insert(0, current_label)
        
        # Update status label
        if current_label:
            self.current_label_status.config(
                text=f"✓ Nhãn hiện tại: '{current_label}' (Có thể chỉnh sửa)",
                fg="#008800"
            )
        else:
            self.current_label_status.config(
                text="✗ Chưa có nhãn - Nhập nhãn mới",
                fg="#cc6600"
            )
        
        # Stop current video if playing
        self.labeling_video_playing = False
        
        # Play the original video (_orig version)
        out_dir = 'gait/preprocess'
        orig_filename = filename.replace('.mp4', '_orig.mp4')
        video_path = os.path.join(out_dir, orig_filename)
        
        if os.path.exists(video_path):
            # Update current video and start new thread
            self.current_labeling_video = video_path
            self.labeling_video_playing = True
            threading.Thread(target=self._play_labeling_video, args=(video_path,), daemon=True).start()
        else:
            self.status_label.config(text=f"Original video not found: {orig_filename}")
    
    def _play_labeling_video(self, video_path):
        """Play video on loop for labeling preview"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        while self.labeling_video_playing and self.current_labeling_video == video_path:
            ret, frame = cap.read()
            if not ret:
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize to larger display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Scale up to larger size for better visibility
            h, w = frame.shape[:2]
            scale = 3.0  # Scale up 3x
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            # Update display (thread-safe)
            try:
                if self.labeling_video_playing and self.current_labeling_video == video_path:
                    self.video_label.config(image=photo)
                    self.video_label.image = photo  # Keep reference
                else:
                    break
            except:
                break
            
            # Check if still on labeling tab (index 5 after reorder)
            if self.current_tab != 5:
                break
            
            cv2.waitKey(30)  # ~30 fps
        
        cap.release()
    
    def save_label(self):
        """Save label for selected file"""
        # Sử dụng biến instance thay vì curselection()
        if not self.selected_labeling_file:
            self.status_label.config(text="Không có file nào được chọn")
            return
        
        filename = self.selected_labeling_file
        new_label = self.label_entry.get().strip()
        
        # Allow empty label (to clear/remove label)
        # if not new_label:
        #     self.status_label.config(text="Nhãn không được để trống")
        #     return
        
        # Check if label is being updated
        old_label = self.gait_labels.get(filename, "")
        is_update = bool(old_label)
        
        # Save label to JSON
        self.gait_labels[filename] = new_label
        self._save_gait_labels()
        
        # Update status message
        if is_update:
            if old_label == new_label:
                self.status_label.config(text=f"✓ Nhãn '{new_label}' đã được xác nhận")
            else:
                self.status_label.config(text=f"✓ Đã cập nhật nhãn: '{old_label}' → '{new_label}'")
                self.current_label_status.config(
                    text=f"✓ Nhãn hiện tại: '{new_label}' (Có thể chỉnh sửa)",
                    fg="#008800"
                )
        else:
            self.status_label.config(text=f"✓ Đã lưu nhãn '{new_label}' cho {filename}")
            self.current_label_status.config(
                text=f"✓ Nhãn hiện tại: '{new_label}' (Có thể chỉnh sửa)",
                fg="#008800"
            )
        
        # Refresh list to show updated label
        self.refresh_labeling_list()
        
        # Restore selection by finding the file
        for i in range(self.labeling_file_listbox.size()):
            display_text = self.labeling_file_listbox.get(i)
            if filename in display_text:
                self.labeling_file_listbox.selection_set(i)
                self.labeling_file_listbox.see(i)
                break
    
    def generate_gei_all(self):
        """Generate GEI for all labeled files with smart checking (new/changed/unchanged)"""
        out_dir = 'gait/preprocess'
        if not os.path.exists(out_dir):
            self.status_label.config(text="No files to process")
            return
        
        # Get all files that have labels
        files = [f for f in os.listdir(out_dir) if f.endswith('.mp4') and '_orig' not in f]
        labeled_files = [f for f in files if f in self.gait_labels and self.gait_labels[f]]
        
        if not labeled_files:
            self.status_label.config(text="No labeled files to process!")
            return
        
        self.status_label.config(text=f"Checking {len(labeled_files)} files...")
        
        # Run in thread to avoid blocking UI
        threading.Thread(target=self._generate_gei_smart, args=(labeled_files,), daemon=True).start()
    
    def _generate_gei_smart(self, files):
        """Smart GEI generation: check Generated.json and only process new/changed files"""
        try:
            # Load Generated.json to check previous state
            generated_data = self._load_generated_data()
            
            stats = {'new': 0, 'changed': 0, 'skipped': 0, 'total_geis': 0}
            
            for i, filename in enumerate(files, 1):
                current_label = self.gait_labels[filename]
                video_path = os.path.join('gait/preprocess', filename)
                
                # Update progress
                self.root.after(0, lambda i=i, n=len(files), f=filename: 
                    self.status_label.config(text=f"Processing {i}/{n}: {f}"))
                
                # Check status
                if filename in generated_data:
                    old_label = generated_data[filename]
                    if old_label == current_label:
                        # Same label - skip
                        stats['skipped'] += 1
                        continue
                    else:
                        # Label changed - delete old GEI files
                        old_gei_dir = os.path.join('gait/GEI', old_label)
                        if os.path.exists(old_gei_dir):
                            base = os.path.splitext(filename)[0]
                            pattern = f"{base}_part"
                            for f in os.listdir(old_gei_dir):
                                if f.startswith(pattern):
                                    try:
                                        os.remove(os.path.join(old_gei_dir, f))
                                    except Exception:
                                        pass
                        stats['changed'] += 1
                else:
                    # New file
                    stats['new'] += 1
                
                # Generate GEI
                output_dir = os.path.join('gait/GEI', current_label)
                os.makedirs(output_dir, exist_ok=True)
                
                paths = process_video_to_gei(video_path, output_dir, frames_per_gei=30, overwrite=True)
                stats['total_geis'] += len(paths)
                
                # Update Generated.json entry
                generated_data[filename] = current_label
            
            # Save updated Generated.json
            self._save_generated_data(generated_data)
            
            # Show summary
            summary = f"✓ Done: {stats['new']} new, {stats['changed']} changed, {stats['skipped']} skipped | {stats['total_geis']} GEIs"
            self.root.after(0, lambda: self.status_label.config(text=summary, fg="#00ff00"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error: {str(e)}", fg="#ff0000"))
    
    def regenerate_all_gei(self):
        """Regenerate GEI for all preprocessed files (clear and rebuild)"""
        out_dir = 'gait/preprocess'
        if not os.path.exists(out_dir):
            self.status_label.config(text="No files to process")
            return
        
        # Get all files that have labels
        files = [f for f in os.listdir(out_dir) if f.endswith('.mp4') and '_orig' not in f]
        labeled_files = [f for f in files if f in self.gait_labels and self.gait_labels[f]]
        
        if not labeled_files:
            self.status_label.config(text="No labeled files to process!")
            return
        
        # Confirm action
        from tkinter import messagebox
        confirm = messagebox.askyesno(
            "Regenerate All GEI",
            f"This will DELETE all existing GEI files and regenerate for {len(labeled_files)} labeled files.\n\nContinue?"
        )
        
        if not confirm:
            return
        
        self.status_label.config(text=f"Regenerating GEI for {len(labeled_files)} files...")
        
        # Run in thread
        threading.Thread(target=self._regenerate_all_gei_thread, args=(labeled_files,), daemon=True).start()
    
    def _regenerate_all_gei_thread(self, files):
        """Regenerate all GEI files in background"""
        try:
            # Clear all GEI directory
            gei_root = 'gait/GEI'
            if os.path.exists(gei_root):
                shutil.rmtree(gei_root)
            os.makedirs(gei_root, exist_ok=True)
            
            total_geis = 0
            generated_data = {}
            
            for i, filename in enumerate(files, 1):
                label = self.gait_labels[filename]
                video_path = os.path.join('gait/preprocess', filename)
                output_dir = os.path.join(gei_root, label)
                os.makedirs(output_dir, exist_ok=True)
                
                # Update status
                self.root.after(0, lambda i=i, n=len(files), f=filename: 
                    self.status_label.config(text=f"Processing {i}/{n}: {f}"))
                
                # Generate GEI
                paths = process_video_to_gei(video_path, output_dir, frames_per_gei=30, overwrite=True)
                total_geis += len(paths)
                
                # Track in generated data
                generated_data[filename] = label
            
            # Save Generated.json
            self._save_generated_data(generated_data)
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"✓ Regenerated {total_geis} GEI images from {len(files)} videos", fg="#00ff00"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error: {str(e)}", fg="#ff0000"))
    
    def _load_gait_labels(self):
        """Load gait labels from JSON file"""
        if os.path.exists(self.gait_labels_file):
            try:
                with open(self.gait_labels_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading gait labels: {e}")
                return {}
        return {}
    
    def _save_gait_labels(self):
        """Save gait labels to JSON file"""
        os.makedirs(os.path.dirname(self.gait_labels_file), exist_ok=True)
        try:
            with open(self.gait_labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.gait_labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving gait labels: {e}")
    
    def _load_generated_data(self):
        """Load Generated.json tracking which files have been generated"""
        if os.path.exists(self.generated_file):
            try:
                with open(self.generated_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading generated data: {e}")
                return {}
        return {}
    
    def _save_generated_data(self, data):
        """Save Generated.json"""
        os.makedirs(os.path.dirname(self.generated_file), exist_ok=True)
        try:
            with open(self.generated_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving generated data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
