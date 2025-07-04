"""
Video processing modules for SyncTalk 2D.

This submodule contains the main video generation processors:
- MediaProcessor: Unified media processing (frame extraction, landmarks, audio)
- VideoProcessor: Alias for MediaProcessor (backward compatibility)
- StandardVideoProcessor: Standard mode video generation using pre-extracted frames
- CoreClipsProcessor: Core clips mode using pre-recorded video segments
"""

from .media_processor import MediaProcessor
from .standard_processor import StandardVideoProcessor
from .core_clips_processor import CoreClipsProcessor

# Backward compatibility alias
VideoProcessor = MediaProcessor

__all__ = ['MediaProcessor', 'VideoProcessor', 'StandardVideoProcessor', 'CoreClipsProcessor']