"""
Utility functions for SyncTalk 2D.

This submodule contains shared utilities:
- Face blending functions for seamless integration
- Image processing utilities
- Helper functions for video generation
"""

from .face_blending import (
    create_face_mask,
    blend_faces,
    match_color_histogram,
    get_face_region_with_padding,
    align_landmarks_to_reference
)

__all__ = [
    'create_face_mask',
    'blend_faces',
    'match_color_histogram',
    'get_face_region_with_padding',
    'align_landmarks_to_reference'
]