import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import the frame-based structures
from frame_based_structures import FrameBasedClipSelection, CoreClip


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
        
        # Track frame usage for each clip to enable cycling
        self.clip_frame_positions = {}
        
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
            
            # Determine clip type from filename
            if "silence" in filename:
                clip_type = "silence"
            elif "talk" in filename:
                clip_type = "talk"
            else:
                continue  # Skip unrecognized files
                
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
                    with open(info_file, 'r') as f:
                        for line in f:
                            if line.startswith("FPS:"):
                                fps = float(line.split(":")[1].strip())
                                break
                
                duration = frame_count / fps if frame_count > 0 else 0.0
            else:
                print(f"Warning: No preprocessed data found for {video_file.name}")
                continue
                
            if frame_count == 0:
                print(f"Warning: No frames found for {video_file.name}")
                continue
                
            clip = CoreClip(str(video_file), clip_type, duration)
            clip.frame_count = frame_count
            clip.fps = fps
            
            if clip_type == "talk":
                self.talk_clips.append(clip)
            else:
                self.silence_clips.append(clip)
                
        # Sort clips by frame count for easier selection
        self.talk_clips.sort(key=lambda x: x.frame_count)
        self.silence_clips.sort(key=lambda x: x.frame_count)
        
        print(f"Loaded {len(self.talk_clips)} talk clips and {len(self.silence_clips)} silence clips for {self.model_name}")
        
        # Print loaded clips for debugging
        print("\nTalk clips (sorted by frame count):")
        for clip in self.talk_clips:
            clip_name = Path(clip.path).stem
            print(f"  - {clip_name}: {clip.frame_count} frames ({clip.duration:.2f}s)")
        
        print("\nSilence clips (sorted by frame count):")
        for clip in self.silence_clips:
            clip_name = Path(clip.path).stem
            print(f"  - {clip_name}: {clip.frame_count} frames ({clip.duration:.2f}s)")
        
    def select_clips_for_segment(self, duration: float, clip_type: str, fps: int = 25) -> List[Tuple[CoreClip, int, int, int]]:
        """
        Select clips to fill a segment of given duration using frame-based approach.
        This method cycles through clips to use their entire frame range.
        
        Args:
            duration: Duration to fill (in seconds)
            clip_type: Type of clips to use ("talk" or "silence")
            fps: Frames per second (default: 25)
            
        Returns:
            List of tuples (clip, start_frame, end_frame, padding_frames)
            - start_frame: Starting frame in the clip
            - end_frame: Ending frame in the clip
            - padding_frames: Number of times to repeat the last frame
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
        
        # Strategy: If segment is longer than any single clip, cycle through clips
        # Otherwise, use the best matching clip
        
        # No longer need to find longest clip here - we'll find best match directly
        
        # Find the best matching clip
        # Strategy: Prefer clips that can play fully without being cut
        best_clip = None
        best_score = float('inf')
        best_strategy = None  # "single", "multiple", or "pad"
        
        # Debug: print segment info
        print(f"\n[DEBUG] Selecting clip for {clip_type} segment of {remaining_frames} frames ({remaining_frames/fps:.2f}s)")
        print(f"Available clips:")
        for clip in valid_clips:
            clip_name = Path(clip.path).stem
            print(f"  - {clip_name}: {clip.frame_count} frames ({clip.frame_count/fps:.2f}s)")
        
        # Strategy: Fill segments efficiently by combining clips
        # We'll try different combinations to minimize padding
        
        # First, check if any single clip matches exactly or is very close
        for clip in valid_clips:
            clip_frames = clip.frame_count
            clip_name = Path(clip.path).stem
            
            if clip_frames == remaining_frames:
                # Perfect match!
                best_clip = clip
                best_strategy = "single"
                print(f"\n  Perfect match: {clip_name} ({clip_frames} frames)")
                break
            elif clip_frames > remaining_frames:
                # This clip is longer - we'd need to cut it
                # Only consider if padding would be excessive otherwise
                continue
            else:
                # This clip is shorter - calculate padding needed
                padding_needed = remaining_frames - clip_frames
                # Allow padding up to 100% of the clip length
                padding_ratio = padding_needed / clip_frames  # Changed: ratio relative to clip length, not segment
                
                if padding_ratio <= 1.0:  # Padding up to 100% of clip length
                    # Score based on number of segments needed (prefer fewer segments)
                    score = 1000 + padding_needed  # Prioritize single clip over multiple
                    print(f"\n  Considering {clip_name}: {clip_frames} frames + {padding_needed} padding (ratio: {padding_ratio:.1%} of clip)")
                    
                    if score < best_score:
                        best_score = score
                        best_clip = clip
                        best_strategy = "single"
        
        # If no good single clip found, try combinations
        if best_clip is None or best_score > 2000:  # Only try combinations if no reasonable single clip
            print("\n  No good single clip match, trying combinations...")
            
            # Sort clips by frame count for combination building
            sorted_clips = sorted(valid_clips, key=lambda c: c.frame_count, reverse=True)
            
            # Try to fill with multiple clips
            best_combination = []
            frames_left = remaining_frames
            
            for clip in sorted_clips:
                clip_frames = clip.frame_count
                clip_name = Path(clip.path).stem
                
                if clip_frames <= frames_left:
                    # See how many times we can use this clip
                    num_uses = frames_left // clip_frames
                    frames_used = num_uses * clip_frames
                    frames_left -= frames_used
                    
                    if num_uses > 0:
                        best_combination.extend([clip] * num_uses)
                        print(f"    Using {clip_name} x{num_uses} ({frames_used} frames)")
                    
                    if frames_left == 0:
                        break
            
            # Check if combination is better than single clip with padding
            if best_combination and frames_left < best_score:
                best_clip = best_combination[0]  # Will be processed later
                best_strategy = "combination"
                best_score = frames_left
                print(f"    Remaining to pad: {frames_left} frames")
        
        if best_clip is None:
            # Fallback to the clip with most frames
            best_clip = max(valid_clips, key=lambda c: c.frame_count)
            best_strategy = "single"
        
        # Get frame count for the selected clip
        clip_total_frames = best_clip.frame_count
        
        # Now create the selection based on strategy
        if best_strategy == "combination":
            # Re-build the combination properly
            frames_left = remaining_frames
            sorted_clips = sorted(valid_clips, key=lambda c: c.frame_count, reverse=True)
            
            for clip in sorted_clips:
                clip_frames = clip.frame_count
                
                while clip_frames <= frames_left:
                    selected.append((clip, 0, clip_frames, 0))
                    frames_left -= clip_frames
                    
                    if frames_left == 0:
                        break
                
                if frames_left == 0:
                    break
            
            # If there are still frames left, pad the last clip
            if frames_left > 0 and selected:
                # Remove the last entry and re-add with padding
                last_clip, start, end, _ = selected[-1]
                selected[-1] = (last_clip, start, end, frames_left)
                
        elif best_strategy == "single":
            # Single clip with possible padding
            padding_frames = remaining_frames - clip_total_frames
            selected.append((best_clip, 0, clip_total_frames, padding_frames))
            if padding_frames > 0:
                print(f"\n  Selected: {Path(best_clip.path).stem} with {padding_frames} frames padding")
            else:
                print(f"\n  Selected: {Path(best_clip.path).stem} (perfect fit)")
                
        elif best_strategy == "cut":
            # Clip is longer than segment - use only what we need
            selected.append((best_clip, 0, remaining_frames, 0))
            print(f"\n  Selected: {Path(best_clip.path).stem} - using first {remaining_frames} frames only")
            
        else:
            # Fallback - shouldn't reach here
            padding_frames = max(0, remaining_frames - clip_total_frames)
            selected.append((best_clip, 0, min(clip_total_frames, remaining_frames), padding_frames))
            print(f"\n  Selected (fallback): {Path(best_clip.path).stem}")
                
        return selected
        
    def select_clips_for_segment_v2(self, duration: float, clip_type: str, 
                                    output_start_frame: int = 0, fps: int = 25) -> List[FrameBasedClipSelection]:
        """
        Select clips to fill a segment using FrameBasedClipSelection objects.
        
        Args:
            duration: Duration to fill (in seconds)
            clip_type: Type of clips to use ("talk" or "silence")
            output_start_frame: Starting frame in the output timeline
            fps: Frames per second (default: 25)
            
        Returns:
            List of FrameBasedClipSelection objects
        """
        clips = self.talk_clips if clip_type == "talk" else self.silence_clips
        
        if not clips:
            raise ValueError(f"No {clip_type} clips available")
            
        # Filter out clips with invalid duration
        valid_clips = [c for c in clips if c.duration > 0]
        if not valid_clips:
            raise ValueError(f"No valid {clip_type} clips with duration > 0")
            
        selections = []
        total_frames_needed = int(duration * fps)
        remaining_frames = total_frames_needed
        current_output_frame = output_start_frame
        
        # Select the best clip based on duration match
        best_clip = valid_clips[-1]  # Default to longest
        best_score = float('inf')
        
        for clip in valid_clips:
            clip_frames = int(clip.duration * fps)
            # Prefer clips that require less padding
            if clip_frames <= remaining_frames:
                score = remaining_frames - clip_frames
                if score < best_score:
                    best_score = score
                    best_clip = clip
        
        # Get actual frame count from preprocessed data
        clip_name = Path(best_clip.path).stem
        preprocessed_frames_dir = self.preprocessed_dir / clip_name / "full_body_img"
        actual_frame_count = len(list(preprocessed_frames_dir.glob("*.jpg")))
        
        if actual_frame_count == 0:
            # Fallback to calculated frames
            actual_frame_count = int(best_clip.duration * fps)
        
        if actual_frame_count >= remaining_frames:
            # Clip is longer than needed, use only what we need
            selection = FrameBasedClipSelection(
                clip=best_clip,
                start_frame=0,
                end_frame=remaining_frames,
                output_start_frame=current_output_frame,
                output_end_frame=current_output_frame + remaining_frames,
                padding_frames=0
            )
        else:
            # Clip is shorter, play it fully and pad the rest
            padding_frames = remaining_frames - actual_frame_count
            selection = FrameBasedClipSelection(
                clip=best_clip,
                start_frame=0,
                end_frame=actual_frame_count,
                output_start_frame=current_output_frame,
                output_end_frame=current_output_frame + remaining_frames,
                padding_frames=padding_frames
            )
        
        selections.append(selection)
        return selections
        
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
        
    def get_clip_info(self) -> Dict:
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