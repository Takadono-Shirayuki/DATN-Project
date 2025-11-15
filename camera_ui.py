"""
Simple Camera UI using tkinter
"""
import tkinter as tk
from tkinter import ttk
import threading
import cv2
from PIL import Image, ImageTk
from camera_module import CameraModule
from object_box_module import ObjectBoxModule
import os

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Form1")
        self.root.geometry("1400x750")
        
        # Variables
        self.webcam_ip = tk.StringVar()
        self.loaded = False
        self.running = False
        self.current_tab = 0  # 0: Camera, 1: Object Box, 2: Recorder
        self.update_thread = None
        
        # FPS tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = None
        self.current_resolution = "N/A"
        
        # Modules
        self.camera_module = CameraModule()
        self.objectbox_module = ObjectBoxModule()
        
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
        
        # Webcam input (for Camera tab)
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
        
        tk.Checkbutton(self.left_panel_camera, text="Hộp giới hạn", variable=self.bbox_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_camera, text="Khung xương", variable=self.pose_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        tk.Checkbutton(self.left_panel_camera, text="Phân đoạn ảnh", variable=self.seg_var,
                      font=("Segoe UI", 12), command=self.on_checkbox_change).pack(anchor=tk.W, padx=15, pady=5)
        
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
        
        # Right panel (video display)
        right_panel = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2, bg="black")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(right_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Hide all panels
        self.left_panel_camera.pack_forget()
        self.left_panel_objectbox.pack_forget()
        self.left_panel_recorder.pack_forget()
        
        if tab_index == 0:  # Camera tab
            self.camera_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_camera.pack(fill=tk.BOTH, expand=True)
            
            # Show webcam controls
            self.webcam_frame.pack(side=tk.LEFT, padx=20)
            
            self.status_label.config(text="Camera tab selected")
            
        elif tab_index == 1:  # Object Box tab
            self.objectbox_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_objectbox.pack(fill=tk.BOTH, expand=True)
            
            # Show webcam controls
            self.webcam_frame.pack(side=tk.LEFT, padx=20)
            
            self.status_label.config(text="Object Box tab selected")
            
        else:  # Recorder tab
            self.recorder_tab_btn.config(font=("Segoe UI", 12, "bold"), relief=tk.SUNKEN)
            self.left_panel_recorder.pack(fill=tk.BOTH, expand=True)
            
            # Hide webcam controls
            self.webcam_frame.pack_forget()
            
            self.status_label.config(text="Recorder tab selected")
            
            # Load video list
            self.load_video_list()
    
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
                segmentation=self.seg_var.get()
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
                segmentation=self.seg_var.get()
            )
        elif self.current_tab == 1:  # Object Box mode
            self.objectbox_module.set_options(
                pose=self.obj_pose_var.get(),
                segmentation=self.obj_seg_var.get()
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
