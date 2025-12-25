"""
Simple Camera UI using tkinter
"""
import tkinter as tk
import threading
import cv2
from PIL import Image, ImageTk
from camera_tab.camera_module import CameraModule
from object_box_tab.object_box_module import ObjectBoxModule
from dataset_tab.dataset_manager import DatasetManager
import os

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Form1")
        self.root.geometry("1400x750")
        
        # Variables
        self.webcam_ip = tk.StringVar()
        self.dataset_path = tk.StringVar()  # Custom dataset path
        self.loaded = False
        self.running = False
        self.current_tab = 0  # 0: Camera, 1: Object Box, 2: Recorder, 3: Dataset
        self.update_thread = None
        
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
        
        self.recorder_tab_btn = tk.Button(tab_button_frame, text="Recorder", width=15,
                                           font=("Segoe UI", 12),
                                           command=lambda: self.switch_tab(2))
        self.recorder_tab_btn.pack(side=tk.LEFT, padx=2)
        
        self.dataset_tab_btn = tk.Button(tab_button_frame, text="Dataset", width=15,
                                          font=("Segoe UI", 12),
                                          command=lambda: self.switch_tab(3))
        self.dataset_tab_btn.pack(side=tk.LEFT, padx=2)
        
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
        
        # Left panel - Recorder tab content (initially hidden)
        self.left_panel_recorder = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_recorder.pack_propagate(False)
        
        # Recorder controls
        tk.Label(self.left_panel_recorder, text="Recorder  Hộp đối tượng", 
                font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # Checkboxes for Recorder
        self.rec_bbox_var = tk.BooleanVar()
        self.rec_pose_var = tk.BooleanVar()
        self.rec_seg_var = tk.BooleanVar()
        
        tk.Checkbutton(self.left_panel_recorder, text="Hộp giới hạn", variable=self.rec_bbox_var,
                      font=("Segoe UI", 10)).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_recorder, text="Khung xương", variable=self.rec_pose_var,
                      font=("Segoe UI", 10)).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_recorder, text="Phân đoạn ảnh", variable=self.rec_seg_var,
                      font=("Segoe UI", 10)).pack(anchor=tk.W, padx=15, pady=5)
        
        # Video selection listbox
        tk.Label(self.left_panel_recorder, text="Danh sách video:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(20,5), anchor=tk.W, padx=10)
        
        self.video_listbox = tk.Listbox(self.left_panel_recorder, height=10, font=("Segoe UI", 9))
        self.video_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Play button for Recorder
        tk.Button(self.left_panel_recorder, text="Play", width=15, 
                 font=("Segoe UI", 10)).pack(pady=15)
        
        # Left panel - Dataset tab content (initially hidden)
        self.left_panel_dataset = tk.Frame(self.left_container, relief=tk.RAISED, borderwidth=1)
        self.left_panel_dataset.pack_propagate(False)
        
        # Dataset controls
        tk.Label(self.left_panel_dataset, text="Dataset Generator", 
                font=("Segoe UI", 11, "bold")).pack(pady=15, anchor=tk.W, padx=10)
        
        # Note: Pose and Segmentation are always enabled for dataset generation
        tk.Label(self.left_panel_dataset, text="✓ Pose keypoints enabled\n✓ Segmentation mask enabled", 
                font=("Segoe UI", 9), fg="green", justify=tk.LEFT).pack(anchor=tk.W, padx=15, pady=5)
        
        # File selection
        tk.Label(self.left_panel_dataset, text="Video files:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(20,5), anchor=tk.W, padx=10)
        
        # File listbox
        file_frame = tk.Frame(self.left_panel_dataset)
        file_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(file_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(file_frame, height=8, font=("Segoe UI", 9),
                                       yscrollcommand=scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Buttons
        btn_frame = tk.Frame(self.left_panel_dataset)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Add Files", width=12, 
                 font=("Segoe UI", 10), command=self.add_dataset_files).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Clear", width=12, 
                 font=("Segoe UI", 10), command=self.clear_dataset_files).pack(side=tk.LEFT, padx=2)
        
        # Process button
        self.process_dataset_btn = tk.Button(self.left_panel_dataset, text="Process Dataset", 
                                            width=20, font=("Segoe UI", 10, "bold"),
                                            command=self.process_dataset)
        self.process_dataset_btn.pack(pady=15)
        
        # Dataset management section
        tk.Label(self.left_panel_dataset, text="Dataset Management:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(20,5), anchor=tk.W, padx=10)
        
        # Processed videos listbox
        mgmt_frame = tk.Frame(self.left_panel_dataset)
        mgmt_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        mgmt_scrollbar = tk.Scrollbar(mgmt_frame)
        mgmt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.processed_listbox = tk.Listbox(mgmt_frame, height=6, font=("Segoe UI", 8),
                                           selectmode=tk.MULTIPLE,
                                           yscrollcommand=mgmt_scrollbar.set)
        self.processed_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mgmt_scrollbar.config(command=self.processed_listbox.yview)
        
        # Management buttons
        mgmt_btn_frame = tk.Frame(self.left_panel_dataset)
        mgmt_btn_frame.pack(pady=5)
        
        tk.Button(mgmt_btn_frame, text="Delete Selected", width=12, 
                 font=("Segoe UI", 9), command=self.delete_selected_dataset).pack(side=tk.LEFT, padx=2)
        tk.Button(mgmt_btn_frame, text="Delete All", width=12, 
                 font=("Segoe UI", 9), command=self.delete_all_datasets).pack(side=tk.LEFT, padx=2)
        tk.Button(mgmt_btn_frame, text="Refresh", width=12, 
                 font=("Segoe UI", 9), command=self.refresh_dataset_list).pack(side=tk.LEFT, padx=2)
        
        # Stats label
        self.dataset_stats_label = tk.Label(self.left_panel_dataset, text="", 
                                           font=("Segoe UI", 8), fg="gray")
        self.dataset_stats_label.pack(pady=5)
        
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
        self.recorder_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        self.dataset_tab_btn.config(font=("Segoe UI", 12), relief=tk.RAISED)
        
        # Hide all panels
        self.left_panel_camera.pack_forget()
        self.left_panel_objectbox.pack_forget()
        self.left_panel_recorder.pack_forget()
        self.left_panel_dataset.pack_forget()
        
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
            
        elif tab_index == 2:  # Recorder tab
            self.recorder_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_recorder.pack(fill=tk.BOTH, expand=True)
            
            # Hide both webcam and dataset path controls
            self.webcam_frame.pack_forget()
            self.dataset_path_frame.pack_forget()
            
            self.status_label.config(text="Recorder tab selected")
            
            # Load video list
            self.load_video_list()
            
        else:  # Dataset tab (tab_index == 3)
            self.dataset_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_dataset.pack(fill=tk.BOTH, expand=True)
            
            # Hide webcam controls, show dataset path
            self.webcam_frame.pack_forget()
            self.dataset_path_frame.pack(side=tk.LEFT, padx=20)
            
            # Show info overlay if processing
            if self.dataset_processing:
                self.info_overlay_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=700, height=300)
            
            # Refresh dataset list
            self.refresh_dataset_list()
            
            self.status_label.config(text="Dataset tab selected")
    
    def load_video_list(self):
        """Load list of recorded videos"""
        import os
        self.video_listbox.delete(0, tk.END)
        
        # Look for recorded videos in temp folder
        if os.path.exists("temp"):
            videos = [f for f in os.listdir("temp") if f.endswith(".avi")]
            for video in sorted(videos):
                self.video_listbox.insert(tk.END, video)
        
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

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
