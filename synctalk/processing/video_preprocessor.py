"""
Video preprocessing module for SyncTalk 2D.

This module handles video preprocessing tasks including:
- Frame extraction
- Frame rate conversion
- Facial landmark detection
"""

import os
import cv2
import tempfile
import shutil
import subprocess
from typing import Optional, Callable, Tuple
from pathlib import Path

# Import landmark detector - using relative import to avoid circular dependencies
import sys
sys.path.append('./data_utils')
from data_utils.get_landmark import Landmark


class VideoProcessor:
    """Process uploaded videos to extract frames and landmarks."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize video processor.
        
        Args:
            temp_dir: Temporary directory for processing. Created if None.
        """
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="synctalk_")
        else:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.landmarks_dir = os.path.join(self.temp_dir, "landmarks")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.landmarks_dir, exist_ok=True)
        
        self.landmark_detector: Optional[Landmark] = None
        
    def _init_landmark_detector(self) -> None:
        """Initialize landmark detector lazily."""
        if self.landmark_detector is None:
            self.landmark_detector = Landmark()
    
    def extract_frames(self, video_path: str, 
                      progress_callback: Optional[Callable[[int, int, str], None]] = None) -> int:
        """
        Extract frames from video, converting to 25fps if needed.
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Number of frames extracted
            
        Raises:
            ValueError: If video cannot be opened or processed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert to 25fps if needed
        video_to_process = video_path
        if abs(fps - 25.0) > 0.1:  # Not 25fps
            cap.release()
            if progress_callback:
                progress_callback(0, 100, "Converting video to 25fps...")
            
            converted_path = os.path.join(self.temp_dir, "video_25fps.mp4")
            
            # Use subprocess instead of os.system for better error handling
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-nostats',
                '-i', video_path,
                '-vf', 'fps=25',
                '-c:v', 'libx264',
                converted_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Failed to convert video to 25fps: {e.stderr}")
            
            video_to_process = converted_path
            cap = cv2.VideoCapture(converted_path)
            if not cap.isOpened():
                raise ValueError("Failed to open converted video")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        if progress_callback:
            progress_callback(0, total_frames, "Extracting frames...")
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(self.frames_dir, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames, 
                                f"Extracting frame {frame_count}/{total_frames}")
        
        cap.release()
        return frame_count
    
    def detect_landmarks(self, 
                        progress_callback: Optional[Callable[[int, int, str], None]] = None) -> bool:
        """
        Detect landmarks for all extracted frames.
        
        Args:
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            True if successful, False otherwise
        """
        self._init_landmark_detector()
        
        frame_files = sorted(
            [f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )
        
        if not frame_files:
            return False
            
        total_frames = len(frame_files)
        if progress_callback:
            progress_callback(0, total_frames, "Detecting landmarks...")
        
        last_landmarks: Optional[str] = None
        successful_detections = 0
        
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(self.frames_dir, frame_file)
            lms_path = os.path.join(self.landmarks_dir, frame_file.replace('.jpg', '.lms'))
            
            # Detect landmarks
            try:
                pre_landmark, x1, y1 = self.landmark_detector.detect(frame_path)
                
                if pre_landmark is not None:
                    # Save landmarks
                    lms_lines = []
                    for p in pre_landmark:
                        x, y = p[0] + x1, p[1] + y1
                        lms_lines.append(f"{x} {y}")
                    
                    landmarks_content = "\n".join(lms_lines) + "\n"
                    with open(lms_path, "w") as f:
                        f.write(landmarks_content)
                    
                    last_landmarks = landmarks_content
                    successful_detections += 1
                else:
                    # Use last valid landmarks if available
                    if last_landmarks:
                        with open(lms_path, "w") as f:
                            f.write(last_landmarks)
                    else:
                        print(f"Warning: No landmarks detected for frame {idx}")
                        
            except Exception as e:
                print(f"Error detecting landmarks for frame {idx}: {e}")
                # Use last valid landmarks if available
                if last_landmarks:
                    with open(lms_path, "w") as f:
                        f.write(last_landmarks)
            
            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(idx + 1, total_frames, 
                                f"Detecting landmarks {idx + 1}/{total_frames}")
        
        if progress_callback:
            progress_callback(total_frames, total_frames, "Landmark detection complete")
            
        return successful_detections > 0
    
    def process_video(self, video_path: str, 
                     progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[str, str, int]:
        """
        Process video to extract frames and landmarks.
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Tuple of (frames_dir, landmarks_dir, num_frames)
            
        Raises:
            ValueError: If processing fails
        """
        # Extract frames
        num_frames = self.extract_frames(video_path, progress_callback)
        
        if num_frames == 0:
            raise ValueError("No frames extracted from video")
        
        # Detect landmarks
        success = self.detect_landmarks(progress_callback)
        
        if not success:
            raise ValueError("Failed to detect any landmarks in video")
        
        return self.frames_dir, self.landmarks_dir, num_frames
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary files."""
        self.cleanup()