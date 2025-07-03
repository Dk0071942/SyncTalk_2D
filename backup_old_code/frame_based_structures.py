"""Frame-based data structures for core clips processing."""
from dataclasses import dataclass
from typing import Optional


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
    
    def __repr__(self):
        return (f"FrameSelection({self.clip.clip_type}, "
                f"clip_frames={self.start_frame}-{self.end_frame}, "
                f"output_frames={self.output_start_frame}-{self.output_end_frame}, "
                f"padding={self.padding_frames})")