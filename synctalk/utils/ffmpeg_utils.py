"""
FFmpeg utilities and standardized encoding parameters for SyncTalk 2D.

This module provides consistent FFmpeg encoding parameters across the entire codebase.
"""

import subprocess
from typing import List, Optional, Dict, Any


class FFmpegConfig:
    """Centralized FFmpeg configuration for consistent video encoding."""
    
    # Standard video encoding parameters
    VIDEO_CODEC = "libx264"
    PRESET = "slow"
    CRF = "18"
    PIX_FMT = "yuv420p"
    COLOR_SPACE = "bt709"
    COLOR_PRIMARIES = "bt709"
    COLOR_TRC = "bt709"
    
    # Standard flags
    MOVFLAGS = "+faststart"
    
    @classmethod
    def get_encoding_params(cls, include_audio: bool = True) -> List[str]:
        """
        Get standard FFmpeg encoding parameters.
        
        Args:
            include_audio: Whether to include audio codec parameters
            
        Returns:
            List of FFmpeg parameters
        """
        params = [
            "-c:v", cls.VIDEO_CODEC,
            "-preset", cls.PRESET,
            "-crf", cls.CRF,
            "-pix_fmt", cls.PIX_FMT,
            "-vf", f"format={cls.PIX_FMT},colorspace=all={cls.COLOR_SPACE}:iall={cls.COLOR_SPACE}:fast=1",
            "-color_primaries", cls.COLOR_PRIMARIES,
            "-color_trc", cls.COLOR_TRC,
            "-colorspace", cls.COLOR_SPACE,
            "-movflags", cls.MOVFLAGS
        ]
        
        if include_audio:
            # Copy audio without re-encoding
            params.extend(["-c:a", "copy"])
        
        return params
    
    @classmethod
    def get_simple_encoding_params(cls) -> List[str]:
        """
        Get simplified encoding parameters for cases where color space is not critical.
        
        Returns:
            List of FFmpeg parameters
        """
        return [
            "-c:v", cls.VIDEO_CODEC,
            "-preset", cls.PRESET,
            "-crf", cls.CRF,
            "-pix_fmt", cls.PIX_FMT,
            "-movflags", cls.MOVFLAGS
        ]


def encode_video(input_path: str, output_path: str, 
                 additional_params: Optional[List[str]] = None,
                 use_simple_params: bool = False) -> subprocess.CompletedProcess:
    """
    Encode a video using standardized FFmpeg parameters.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        additional_params: Additional FFmpeg parameters to include
        use_simple_params: Use simplified encoding parameters
        
    Returns:
        Completed process result
    """
    cmd = ["ffmpeg", "-i", input_path]
    
    # Add any additional input parameters
    if additional_params:
        cmd.extend(additional_params)
    
    # Add standard encoding parameters
    if use_simple_params:
        cmd.extend(FFmpegConfig.get_simple_encoding_params())
    else:
        cmd.extend(FFmpegConfig.get_encoding_params())
    
    # Add output path
    cmd.append(output_path)
    
    # Run FFmpeg
    return subprocess.run(cmd, capture_output=True, text=True)


def convert_fps(input_path: str, output_path: str, target_fps: int = 25) -> subprocess.CompletedProcess:
    """
    Convert video to target FPS using standardized encoding.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_fps: Target frame rate
        
    Returns:
        Completed process result
    """
    cmd = [
        "ffmpeg", "-i", input_path,
        "-r", str(target_fps)
    ]
    
    # Add standard encoding parameters
    cmd.extend(FFmpegConfig.get_encoding_params())
    
    # Add output
    cmd.append(output_path)
    
    return subprocess.run(cmd, capture_output=True, text=True)


def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> subprocess.CompletedProcess:
    """
    Merge audio and video using standardized encoding.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path to output file
        
    Returns:
        Completed process result
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0"
    ]
    
    # Add standard encoding parameters
    cmd.extend(FFmpegConfig.get_encoding_params(include_audio=False))
    
    # Copy audio without re-encoding
    cmd.extend(["-c:a", "copy"])
    
    # Add output
    cmd.append(output_path)
    
    return subprocess.run(cmd, capture_output=True, text=True)


def extract_audio(video_path: str, audio_path: str, sample_rate: int = 16000) -> subprocess.CompletedProcess:
    """
    Extract audio from video.
    
    Args:
        video_path: Path to video file
        audio_path: Path to output audio file
        sample_rate: Target sample rate
        
    Returns:
        Completed process result
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",  # No video
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        "-c:a", "pcm_s16le",  # WAV format
        audio_path
    ]
    
    return subprocess.run(cmd, capture_output=True, text=True)


def concat_videos(video_list: List[str], output_path: str, re_encode: bool = True) -> subprocess.CompletedProcess:
    """
    Concatenate multiple videos.
    
    Args:
        video_list: List of video paths to concatenate
        output_path: Path to output file
        re_encode: Whether to re-encode (True) or copy streams (False)
        
    Returns:
        Completed process result
    """
    # Create concat file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video in video_list:
            f.write(f"file '{video}'\n")
        concat_file = f.name
    
    try:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file
        ]
        
        if re_encode:
            cmd.extend(FFmpegConfig.get_encoding_params())
        else:
            cmd.extend(["-c", "copy"])
        
        cmd.append(output_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        import os
        os.unlink(concat_file)
    
    return result