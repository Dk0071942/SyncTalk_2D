"""
Core clips management module for SyncTalk 2D.

This module manages the library of pre-recorded video clips used in the
Core Clips mode, providing intelligent selection and frame extraction.
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict

from .structures import CoreClip, FrameBasedClipSelection


@dataclass
class ClipMetadata:
    """Cached metadata for a core clip."""
    path: str
    clip_type: str
    duration: float
    frame_count: int
    fps: float
    file_size: int
    last_modified: float


class CoreClipsManager:
    """Manages core video clips for dynamic video generation."""
    
    def __init__(self, model_name: str, cache_metadata: bool = True):
        """
        Initialize the core clips manager.
        
        Args:
            model_name: Name of the model (e.g., "LS1")
            cache_metadata: Whether to cache clip metadata for faster loading
        """
        self.model_name = model_name
        self.clips_dir = Path(f"./core_clips/{model_name}")
        self.preprocessed_dir = Path(f"./dataset/{model_name}/core_clips")
        self.cache_metadata = cache_metadata
        self.metadata_cache_file = self.clips_dir / ".clip_metadata_cache.json"
        
        self.talk_clips: List[CoreClip] = []
        self.silence_clips: List[CoreClip] = []
        
        # Track frame usage for each clip to enable cycling
        self.clip_frame_positions: Dict[str, int] = {}
        
        # Load clips
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
            Duration in seconds (0.0 if no pattern found)
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
    
    def _load_metadata_cache(self) -> Optional[Dict[str, ClipMetadata]]:
        """Load cached metadata if available and valid."""
        if not self.cache_metadata or not self.metadata_cache_file.exists():
            return None
            
        try:
            with open(self.metadata_cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Convert back to ClipMetadata objects
            metadata = {}
            for path, data in cache_data.items():
                metadata[path] = ClipMetadata(**data)
                
            return metadata
        except Exception as e:
            print(f"Warning: Failed to load metadata cache: {e}")
            return None
    
    def _save_metadata_cache(self, metadata: Dict[str, ClipMetadata]) -> None:
        """Save metadata cache to disk."""
        if not self.cache_metadata:
            return
            
        try:
            # Convert to serializable format
            cache_data = {}
            for path, clip_meta in metadata.items():
                cache_data[path] = asdict(clip_meta)
                
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata cache: {e}")
    
    def _get_clip_metadata(self, video_file: Path) -> Optional[ClipMetadata]:
        """Extract metadata for a single clip."""
        filename = video_file.name.lower()
        
        # Determine clip type from filename
        if "silence" in filename:
            clip_type = "silence"
        elif "talk" in filename:
            clip_type = "talk"
        else:
            return None  # Skip unrecognized files
            
        # Get actual frame count and duration from preprocessed data
        clip_name = video_file.stem
        preprocessed_frames_dir = self.preprocessed_dir / clip_name / "full_body_img"
        
        # Count actual frames
        frame_count = 0
        duration = 0.0
        fps = 25  # Default FPS
        
        if preprocessed_frames_dir.exists():
            frame_count = len(list(preprocessed_frames_dir.glob("*.jpg")))
            
            # Also read FPS from info file if available
            info_file = self.preprocessed_dir / clip_name / "info.txt"
            if info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        for line in f:
                            if line.startswith("FPS:"):
                                fps = float(line.split(":")[1].strip())
                                break
                except Exception as e:
                    print(f"Warning: Failed to read info file for {clip_name}: {e}")
            
            duration = frame_count / fps if frame_count > 0 else 0.0
        else:
            print(f"Warning: No preprocessed data found for {video_file.name}")
            return None
            
        if frame_count == 0:
            print(f"Warning: No frames found for {video_file.name}")
            return None
            
        # Get file stats
        file_stats = video_file.stat()
        
        return ClipMetadata(
            path=str(video_file),
            clip_type=clip_type,
            duration=duration,
            frame_count=frame_count,
            fps=fps,
            file_size=file_stats.st_size,
            last_modified=file_stats.st_mtime
        )
        
    def _load_clips(self) -> None:
        """Load and categorize all available core clips."""
        if not self.clips_dir.exists():
            raise ValueError(f"Core clips directory not found: {self.clips_dir}")
        
        # Try to load from cache first
        cached_metadata = self._load_metadata_cache()
        metadata_to_save = {}
        
        for video_file in self.clips_dir.glob("*.mp4"):
            # Check if we have valid cached metadata
            if cached_metadata and str(video_file) in cached_metadata:
                cached = cached_metadata[str(video_file)]
                # Verify the file hasn't changed
                file_stats = video_file.stat()
                if (cached.file_size == file_stats.st_size and 
                    cached.last_modified == file_stats.st_mtime):
                    # Use cached metadata
                    clip = CoreClip(cached.path, cached.clip_type, cached.duration)
                    clip.frame_count = cached.frame_count
                    clip.fps = cached.fps
                    
                    if cached.clip_type == "talk":
                        self.talk_clips.append(clip)
                    else:
                        self.silence_clips.append(clip)
                    
                    metadata_to_save[str(video_file)] = cached
                    continue
            
            # Load metadata for this clip
            clip_meta = self._get_clip_metadata(video_file)
            if clip_meta is None:
                continue
                
            # Create clip object
            clip = CoreClip(clip_meta.path, clip_meta.clip_type, clip_meta.duration)
            clip.frame_count = clip_meta.frame_count
            clip.fps = clip_meta.fps
            
            if clip_meta.clip_type == "talk":
                self.talk_clips.append(clip)
            else:
                self.silence_clips.append(clip)
                
            metadata_to_save[str(video_file)] = clip_meta
        
        # Save updated cache
        if self.cache_metadata:
            self._save_metadata_cache(metadata_to_save)
        
        # Sort clips by frame count for easier selection
        self.talk_clips.sort(key=lambda x: x.frame_count)
        self.silence_clips.sort(key=lambda x: x.frame_count)
        
        print(f"Loaded {len(self.talk_clips)} talk clips and {len(self.silence_clips)} silence clips for {self.model_name}")
        
        # Print loaded clips for debugging
        if self.talk_clips or self.silence_clips:
            print("\nTalk clips (sorted by frame count):")
            for clip in self.talk_clips:
                clip_name = Path(clip.path).stem
                print(f"  - {clip_name}: {clip.frame_count} frames ({clip.duration:.2f}s)")
            
            print("\nSilence clips (sorted by frame count):")
            for clip in self.silence_clips:
                clip_name = Path(clip.path).stem
                print(f"  - {clip_name}: {clip.frame_count} frames ({clip.duration:.2f}s)")
    
    def select_clips_for_segment(self, duration: float, clip_type: str, 
                                fps: int = 25) -> List[Tuple[CoreClip, int, int, int]]:
        """
        Select clips to fill a segment of given duration using intelligent selection.
        
        This method uses various strategies to minimize padding and create
        natural-looking video segments.
        
        Args:
            duration: Duration to fill (in seconds)
            clip_type: Type of clips to use ("talk" or "silence")
            fps: Frames per second (default: 25)
            
        Returns:
            List of tuples (clip, start_frame, end_frame, padding_frames)
            - start_frame: Starting frame in the clip
            - end_frame: Ending frame in the clip
            - padding_frames: Number of times to repeat the last frame
            
        Raises:
            ValueError: If no clips are available
        """
        clips = self.talk_clips if clip_type == "talk" else self.silence_clips
        
        if not clips:
            raise ValueError(f"No {clip_type} clips available")
            
        # Filter out clips with invalid duration
        valid_clips = [c for c in clips if c.duration > 0]
        if not valid_clips:
            raise ValueError(f"No valid {clip_type} clips with duration > 0")
            
        selected = []
        total_frames_needed = int(duration * fps)
        remaining_frames = total_frames_needed
        
        # Debug info
        print(f"\n[DEBUG] Selecting clips for {clip_type} segment: {remaining_frames} frames ({duration:.2f}s)")
        
        # Strategy 1: Find perfect match
        for clip in valid_clips:
            if clip.frame_count == remaining_frames:
                selected.append((clip, 0, clip.frame_count, 0))
                print(f"  → Perfect match: {Path(clip.path).stem}")
                return selected
        
        # Strategy 2: Find clip that requires minimal padding
        best_clip = None
        best_padding = float('inf')
        
        for clip in valid_clips:
            if clip.frame_count < remaining_frames:
                padding = remaining_frames - clip.frame_count
                padding_ratio = padding / clip.frame_count
                
                # Accept padding up to 100% of clip length
                if padding_ratio <= 1.0 and padding < best_padding:
                    best_clip = clip
                    best_padding = padding
        
        if best_clip:
            selected.append((best_clip, 0, best_clip.frame_count, best_padding))
            print(f"  → Selected with padding: {Path(best_clip.path).stem} + {best_padding} frames")
            return selected
        
        # Strategy 3: Use multiple clips to minimize padding
        frames_left = remaining_frames
        sorted_clips = sorted(valid_clips, key=lambda c: c.frame_count, reverse=True)
        
        for clip in sorted_clips:
            while clip.frame_count <= frames_left:
                selected.append((clip, 0, clip.frame_count, 0))
                frames_left -= clip.frame_count
                
                if frames_left == 0:
                    break
            
            if frames_left == 0:
                break
        
        # Add padding to last clip if needed
        if frames_left > 0 and selected:
            last_clip, start, end, _ = selected[-1]
            selected[-1] = (last_clip, start, end, frames_left)
            print(f"  → Multiple clips with {frames_left} frames padding on last")
        elif not selected:
            # Fallback: use longest clip with padding
            longest = max(valid_clips, key=lambda c: c.frame_count)
            padding = remaining_frames - longest.frame_count
            selected.append((longest, 0, longest.frame_count, padding))
            print(f"  → Fallback: {Path(longest.path).stem} + {padding} frames")
        
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
        if not preprocessed_frames_dir.exists():
            raise ValueError(
                f"Preprocessed frames not found for {clip_name}. "
                f"Expected directory: {preprocessed_frames_dir}\n"
                f"Please run: python preprocess_core_clips.py --model {self.model_name}"
            )
            
        if not preprocessed_landmarks_dir.exists():
            raise ValueError(
                f"Preprocessed landmarks not found for {clip_name}. "
                f"Expected directory: {preprocessed_landmarks_dir}\n"
                f"Please run: python preprocess_core_clips.py --model {self.model_name}"
            )
        
        num_frames = len(list(preprocessed_frames_dir.glob("*.jpg")))
        if num_frames == 0:
            raise ValueError(f"No frames found in preprocessed data for {clip_name}")
            
        return str(preprocessed_frames_dir), str(preprocessed_landmarks_dir), num_frames
    
    def get_clip_info(self) -> Dict:
        """
        Get information about all loaded clips.
        
        Returns:
            Dictionary with clip information
        """
        return {
            "model_name": self.model_name,
            "clips_directory": str(self.clips_dir),
            "preprocessed_directory": str(self.preprocessed_dir),
            "talk_clips": [
                {
                    "path": c.path, 
                    "duration": c.duration,
                    "frame_count": c.frame_count,
                    "fps": c.fps
                } for c in self.talk_clips
            ],
            "silence_clips": [
                {
                    "path": c.path, 
                    "duration": c.duration,
                    "frame_count": c.frame_count,
                    "fps": c.fps
                } for c in self.silence_clips
            ],
            "total_talk_clips": len(self.talk_clips),
            "total_silence_clips": len(self.silence_clips),
            "total_talk_duration": sum(c.duration for c in self.talk_clips),
            "total_silence_duration": sum(c.duration for c in self.silence_clips),
            "metadata_cached": self.cache_metadata
        }
    
    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        if self.metadata_cache_file.exists():
            self.metadata_cache_file.unlink()
            print(f"Cleared metadata cache for {self.model_name}")