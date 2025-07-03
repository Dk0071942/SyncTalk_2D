import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class CoreClip:
    """Represents a single core video clip with metadata."""
    
    def __init__(self, path: str, clip_type: str, duration: float):
        self.path = path
        self.clip_type = clip_type  # "talk" or "silence"
        self.duration = duration
        self.frames = None
        self.landmarks = None
        self.fps = 25  # Default FPS
        self.frame_count = 0
        
    def __repr__(self):
        return f"CoreClip({self.clip_type}, {self.duration:.2f}s, {self.path})"


class CoreClipsManager:
    """Manages core video clips for dynamic video generation."""
    
    def __init__(self, model_name: str):
        """
        Initialize the core clips manager.
        
        Args:
            model_name: Name of the model (e.g., "LS1")
        """
        self.model_name = model_name
        self.clips_dir = Path(f"./core_clips/{model_name}")
        self.preprocessed_dir = Path(f"./dataset/{model_name}/core_clips")
        
        self.talk_clips: List[CoreClip] = []
        self.silence_clips: List[CoreClip] = []
        
        self._load_clips()
        
    def _parse_duration_from_filename(self, filename: str) -> float:
        """
        Parse duration from filename.
        Handles patterns like:
        - "02,08s talk.mp4" -> 2.08 seconds
        - "AD2_talk_4s.mp4" -> 4.0 seconds
        - "AD2_silence_2s.mp4" -> 2.0 seconds
        
        Args:
            filename: Filename to parse
            
        Returns:
            Duration in seconds
        """
        # Try to match patterns like "02,08s" or "2,16s"
        match = re.search(r'(\d+),(\d+)s', filename)
        if match:
            seconds = int(match.group(1))
            centiseconds = int(match.group(2))
            return seconds + centiseconds / 100.0
            
        # Try to match patterns like "_4s" or "_2s"
        match = re.search(r'_(\d+)s', filename)
        if match:
            return float(match.group(1))
            
        return 0.0
        
    def _load_clips(self):
        """Load and categorize all available core clips."""
        if not self.clips_dir.exists():
            raise ValueError(f"Core clips directory not found: {self.clips_dir}")
            
        for video_file in self.clips_dir.glob("*.mp4"):
            filename = video_file.name.lower()
            
            # Determine clip type
            if "silence" in filename:
                clip_type = "silence"
            elif "talk" in filename:
                clip_type = "talk"
            else:
                continue  # Skip unrecognized files
                
            # Parse duration
            duration = self._parse_duration_from_filename(filename)
            
            # If duration not found in filename, get it from preprocessed data or video
            if duration == 0:
                # Try to get duration from preprocessed info file
                clip_name = video_file.stem
                info_file = self.preprocessed_dir / clip_name / "info.txt"
                
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        for line in f:
                            if line.startswith("Duration:"):
                                duration = float(line.split(":")[1].strip().rstrip('s'))
                                break
                
                # If still no duration, get it from video file
                if duration == 0:
                    cap = cv2.VideoCapture(str(video_file))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    if duration == 0:
                        print(f"Warning: Could not determine duration for {video_file.name}")
                        continue  # Skip clips with no duration
                
            clip = CoreClip(str(video_file), clip_type, duration)
            
            if clip_type == "talk":
                self.talk_clips.append(clip)
            else:
                self.silence_clips.append(clip)
                
        # Sort clips by duration for easier selection
        self.talk_clips.sort(key=lambda x: x.duration)
        self.silence_clips.sort(key=lambda x: x.duration)
        
        print(f"Loaded {len(self.talk_clips)} talk clips and {len(self.silence_clips)} silence clips for {self.model_name}")
        
    def select_clips_for_segment(self, duration: float, clip_type: str) -> List[Tuple[CoreClip, float, float]]:
        """
        Select clips to fill a segment of given duration.
        
        Args:
            duration: Duration to fill (in seconds)
            clip_type: Type of clips to use ("talk" or "silence")
            
        Returns:
            List of tuples (clip, start_time, end_time) where times are within the clip
        """
        clips = self.talk_clips if clip_type == "talk" else self.silence_clips
        
        if not clips:
            raise ValueError(f"No {clip_type} clips available")
            
        # Filter out clips with invalid duration
        valid_clips = [c for c in clips if c.duration > 0]
        if not valid_clips:
            raise ValueError(f"No valid {clip_type} clips with duration > 0")
            
        selected = []
        remaining_duration = duration
        
        # Strategy: Use longest clips first, then fill gaps with shorter ones
        # Since clips are loopable, we can use any portion
        
        # For now, simple approach: use one clip and loop if needed
        # TODO: Implement more sophisticated selection based on speech patterns
        
        # Select the longest clip that's not too much longer than needed
        best_clip = valid_clips[-1]  # Default to longest
        for clip in valid_clips:
            if clip.duration >= remaining_duration * 0.8:  # 80% threshold
                best_clip = clip
                break
                
        # Calculate how many loops we need
        if best_clip.duration <= 0:
            raise ValueError(f"Selected clip has invalid duration: {best_clip}")
            
        num_loops = int(np.ceil(remaining_duration / best_clip.duration))
        
        for i in range(num_loops):
            start_time = 0.0
            end_time = min(best_clip.duration, remaining_duration)
            selected.append((best_clip, start_time, end_time))
            remaining_duration -= end_time
            
            if remaining_duration <= 0:
                break
                
        return selected
        
    def extract_frames_and_landmarks(self, clip: CoreClip) -> Tuple[str, str, int]:
        """
        Get frames and landmarks from preprocessed data.
        
        Args:
            clip: The clip to get data for
            
        Returns:
            Tuple of (frames_dir, landmarks_dir, num_frames)
            
        Raises:
            ValueError: If preprocessed data not found
        """
        # Get preprocessed data directory
        clip_name = Path(clip.path).stem
        preprocessed_clip_dir = self.preprocessed_dir / clip_name
        preprocessed_frames_dir = preprocessed_clip_dir / "full_body_img"
        preprocessed_landmarks_dir = preprocessed_clip_dir / "landmarks"
        
        # Check if preprocessed data exists
        if not preprocessed_frames_dir.exists() or not preprocessed_landmarks_dir.exists():
            raise ValueError(
                f"Preprocessed data not found for {clip}. "
                f"Please run: python preprocess_core_clips.py --model {self.model_name}"
            )
        
        num_frames = len(list(preprocessed_frames_dir.glob("*.jpg")))
        if num_frames == 0:
            raise ValueError(f"No frames found in preprocessed data for {clip}")
            
        return str(preprocessed_frames_dir), str(preprocessed_landmarks_dir), num_frames
        
    def get_clip_info(self) -> Dict[str, List[Dict]]:
        """
        Get information about all loaded clips.
        
        Returns:
            Dictionary with clip information
        """
        return {
            "talk_clips": [{"path": c.path, "duration": c.duration} for c in self.talk_clips],
            "silence_clips": [{"path": c.path, "duration": c.duration} for c in self.silence_clips],
            "total_talk_duration": sum(c.duration for c in self.talk_clips),
            "total_silence_duration": sum(c.duration for c in self.silence_clips)
        }