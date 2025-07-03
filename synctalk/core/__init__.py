"""
Core components for SyncTalk 2D.

This submodule contains core functionality:
- VAD (Voice Activity Detection) for audio segmentation
- Data structures for video clips and edit decisions
- Core clips management for dynamic video selection
"""

from .vad import SileroVAD, AudioSegment
from .structures import CoreClip, EditDecisionItem, FrameBasedClipSelection
from .clips_manager import CoreClipsManager

__all__ = [
    'SileroVAD',
    'AudioSegment',
    'CoreClip',
    'EditDecisionItem',
    'FrameBasedClipSelection',
    'CoreClipsManager'
]