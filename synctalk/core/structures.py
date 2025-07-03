"""
Core data structures for SyncTalk 2D.

This module contains the essential data structures used throughout the system
for managing video clips, edit decisions, and frame-based selections.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


class CoreClip:
    """Represents a single core video clip with metadata."""
    
    def __init__(self, path: str, clip_type: str, duration: float):
        """
        Initialize a core clip.
        
        Args:
            path: Path to the video clip file
            clip_type: Type of clip ("talk" or "silence")
            duration: Duration of the clip in seconds
        """
        self.path = path
        self.clip_type = clip_type  # "talk" or "silence"
        self.duration = duration
        self.frames = None  # Will be populated when frames are loaded
        self.landmarks = None  # Will be populated when landmarks are loaded
        self.fps = 25  # Default FPS
        self.frame_count = 0
        
    def __repr__(self) -> str:
        return f"CoreClip({self.clip_type}, {self.duration:.2f}s, {self.path})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison for testing."""
        if not isinstance(other, CoreClip):
            return False
        return (self.path == other.path and 
                self.clip_type == other.clip_type and 
                abs(self.duration - other.duration) < 0.01)


@dataclass
class FrameBasedClipSelection:
    """Represents a frame-based selection from a core clip."""
    
    clip: CoreClip
    start_frame: int  # Starting frame in the clip (0-based)
    end_frame: int    # Ending frame in the clip (exclusive)
    output_start_frame: int  # Starting frame in output timeline
    output_end_frame: int    # Ending frame in output timeline
    padding_frames: int = 0  # Number of frames to pad with last frame
    
    @property
    def clip_frame_count(self) -> int:
        """Number of frames from the clip."""
        return self.end_frame - self.start_frame
    
    @property
    def total_output_frames(self) -> int:
        """Total frames in output including padding."""
        return self.output_end_frame - self.output_start_frame
    
    @property
    def needs_padding(self) -> bool:
        """Whether this selection requires padding."""
        return self.padding_frames > 0
    
    def validate(self) -> bool:
        """Validate the selection parameters."""
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            return False
        if self.output_start_frame < 0 or self.output_end_frame <= self.output_start_frame:
            return False
        if self.padding_frames < 0:
            return False
        # Check that output frames match clip frames + padding
        expected_output_frames = self.clip_frame_count + self.padding_frames
        actual_output_frames = self.total_output_frames
        return expected_output_frames == actual_output_frames
    
    def __repr__(self) -> str:
        return (f"FrameSelection({self.clip.clip_type}, "
                f"clip_frames={self.start_frame}-{self.end_frame}, "
                f"output_frames={self.output_start_frame}-{self.output_end_frame}, "
                f"padding={self.padding_frames})")


class EditDecisionItem:
    """Represents a single edit decision for video assembly."""
    
    def __init__(self, 
                 start_time: float,
                 end_time: float,
                 clip: CoreClip,
                 clip_start_frame: int,
                 clip_end_frame: int,
                 padding_frames: int,
                 needs_lipsync: bool,
                 fps: int = 25):
        """
        Initialize an edit decision item.
        
        Args:
            start_time: Start time in output timeline (seconds)
            end_time: End time in output timeline (seconds)
            clip: The core clip to use
            clip_start_frame: Starting frame in source clip
            clip_end_frame: Ending frame in source clip (exclusive)
            padding_frames: Number of frames to pad with last frame
            needs_lipsync: Whether this segment needs lip-sync processing
            fps: Frames per second (default 25)
        """
        self.start_time = start_time  # In output timeline
        self.end_time = end_time
        self.clip = clip
        self.clip_start_frame = clip_start_frame  # In source clip (frame number)
        self.clip_end_frame = clip_end_frame  # In source clip (frame number)
        self.padding_frames = padding_frames  # Number of frames to pad with last frame
        self.needs_lipsync = needs_lipsync
        self.fps = fps
        
    @property
    def duration(self) -> float:
        """Duration of this edit in seconds."""
        return self.end_time - self.start_time
        
    @property
    def total_frames(self) -> int:
        """Total number of frames including padding."""
        return (self.clip_end_frame - self.clip_start_frame) + self.padding_frames
    
    @property
    def output_start_frame(self) -> int:
        """Starting frame in output timeline."""
        return int(self.start_time * self.fps)
    
    @property
    def output_end_frame(self) -> int:
        """Ending frame in output timeline."""
        return int(self.end_time * self.fps)
    
    def validate(self) -> bool:
        """Validate the edit decision parameters."""
        if self.start_time < 0 or self.end_time <= self.start_time:
            return False
        if self.clip_start_frame < 0 or self.clip_end_frame <= self.clip_start_frame:
            return False
        if self.padding_frames < 0:
            return False
        if self.fps <= 0:
            return False
        # Check that timing is consistent
        expected_frames = int((self.end_time - self.start_time) * self.fps)
        actual_frames = self.total_frames
        # Allow for small rounding differences
        return abs(expected_frames - actual_frames) <= 1
        
    def __repr__(self) -> str:
        return (f"EDL({self.start_time:.2f}-{self.end_time:.2f}, "
                f"{self.clip.clip_type}, frames={self.clip_start_frame}-{self.clip_end_frame}, "
                f"pad={self.padding_frames}, lipsync={self.needs_lipsync})")
    
    def __eq__(self, other) -> bool:
        """Equality comparison for testing."""
        if not isinstance(other, EditDecisionItem):
            return False
        return (abs(self.start_time - other.start_time) < 0.01 and
                abs(self.end_time - other.end_time) < 0.01 and
                self.clip == other.clip and
                self.clip_start_frame == other.clip_start_frame and
                self.clip_end_frame == other.clip_end_frame and
                self.padding_frames == other.padding_frames and
                self.needs_lipsync == other.needs_lipsync and
                self.fps == other.fps)