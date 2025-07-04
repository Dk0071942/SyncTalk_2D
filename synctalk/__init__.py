"""
SyncTalk 2D - A modular 2D lip-sync video generation system.

This package provides a clean, modular implementation of the SyncTalk 2D
video generation system with support for both standard and core clips modes.
"""

__version__ = "1.0.0"

# Import main processing classes for easy access
from .processing.media_processor import MediaProcessor
from .processing.standard_processor import StandardVideoProcessor
from .processing.core_clips_processor import CoreClipsProcessor

# Backward compatibility alias
VideoProcessor = MediaProcessor

# Import core components
from .core.vad import SileroVAD, AudioSegment
from .core.structures import CoreClip, EditDecisionItem, FrameBasedClipSelection
from .core.clips_manager import CoreClipsManager

# Import utilities
from .utils.face_blending import (
    create_face_mask,
    blend_faces,
    match_color_histogram,
    get_face_region_with_padding,
    align_landmarks_to_reference
)

__all__ = [
    # Processing
    'MediaProcessor',
    'VideoProcessor',
    'StandardVideoProcessor', 
    'CoreClipsProcessor',
    
    # Core
    'SileroVAD',
    'AudioSegment',
    'CoreClip',
    'EditDecisionItem',
    'FrameBasedClipSelection',
    'CoreClipsManager',
    
    # Utils
    'create_face_mask',
    'blend_faces',
    'match_color_histogram',
    'get_face_region_with_padding',
    'align_landmarks_to_reference'
]