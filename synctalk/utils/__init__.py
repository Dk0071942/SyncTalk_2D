"""
Utility functions for SyncTalk 2D.

This submodule contains shared utilities:
- Face blending functions for seamless integration
- Image processing utilities
- Helper functions for video generation
- Unified video processing for frame extraction and landmarks
- Preprocessing status checking and validation
- Batch processing for multiple videos
"""

from .face_blending import (
    create_face_mask,
    blend_faces,
    match_color_histogram,
    get_face_region_with_padding,
    align_landmarks_to_reference
)

from .video_processor import (
    UnifiedVideoProcessor,
    process_video
)

from .preprocessing_utils import (
    check_preprocessing_status,
    validate_preprocessed_data,
    get_preprocessing_info,
    save_preprocessing_metadata,
    load_preprocessing_metadata
)

from .batch_processing import (
    BatchProcessor,
    create_batch_config_template
)

from .ffmpeg_utils import (
    FFmpegConfig,
    encode_video,
    convert_fps,
    merge_audio_video,
    extract_audio,
    concat_videos
)

from .progress import (
    ProgressBar,
    create_progress_callback,
    wrap_iterable,
    safe_print
)

__all__ = [
    'create_face_mask',
    'blend_faces',
    'match_color_histogram',
    'get_face_region_with_padding',
    'align_landmarks_to_reference',
    'UnifiedVideoProcessor',
    'process_video',
    'check_preprocessing_status',
    'validate_preprocessed_data',
    'get_preprocessing_info',
    'save_preprocessing_metadata',
    'load_preprocessing_metadata',
    'BatchProcessor',
    'create_batch_config_template',
    'FFmpegConfig',
    'encode_video',
    'convert_fps',
    'merge_audio_video',
    'extract_audio',
    'concat_videos',
    'ProgressBar',
    'create_progress_callback',
    'wrap_iterable',
    'safe_print'
]