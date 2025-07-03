"""
Configuration management for SyncTalk 2D.

This module provides dataclasses and utilities for managing
configuration across the system.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""
    
    name: str
    checkpoint_path: str = ""
    dataset_path: str = ""
    device: str = "cuda"
    
    # Model architecture
    input_channels: int = 6
    audio_mode: str = "ave"  # "ave", "hubert", or "wenet"
    
    # Inference settings
    use_parsing: bool = False
    loop_back: bool = True
    start_frame: int = 0
    
    def __post_init__(self):
        """Set default paths if not provided."""
        if not self.checkpoint_path:
            self.checkpoint_path = f"./checkpoint/{self.name}"
        if not self.dataset_path:
            self.dataset_path = f"./dataset/{self.name}"


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    
    # Audio encoder settings
    encoder_checkpoint: str = "model/checkpoints/audio_visual_encoder.pth"
    sample_rate: int = 16000
    mel_bins: int = 80
    
    # VAD settings (for core clips mode)
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    min_silence_duration: float = 0.75  # seconds
    
    # Feature extraction
    batch_size: int = 64
    window_size: int = 16  # frames


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    # Video settings
    fps: int = 25
    default_width: int = 320
    default_height: int = 320
    
    # Face processing
    face_crop_size: int = 328
    face_output_size: int = 320
    face_padding: int = 4
    
    # Mouth masking (for standard mode)
    mouth_mask_x1: int = 5
    mouth_mask_y1: int = 5
    mouth_mask_x2: int = 310
    mouth_mask_y2: int = 305
    
    # Output settings
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20  # Quality factor for x264
    
    # Core clips settings
    core_clips_dir: str = "./core_clips"
    preprocessed_clips_dir: str = "./dataset/{model_name}/core_clips"


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior."""
    
    # Temporary files
    temp_dir_prefix: str = "synctalk_"
    cleanup_temp_files: bool = True
    
    # Progress reporting
    enable_progress: bool = True
    progress_update_frequency: int = 10  # Update every N frames
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Performance
    enable_caching: bool = True
    cache_dir: str = "./.cache/synctalk"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class SyncTalkConfig:
    """Main configuration container for SyncTalk 2D."""
    
    model: ModelConfig
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SyncTalkConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SyncTalkConfig instance
        """
        model_config = ModelConfig(**config_dict.get("model", {}))
        audio_config = AudioConfig(**config_dict.get("audio", {}))
        video_config = VideoConfig(**config_dict.get("video", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        
        return cls(
            model=model_config,
            audio=audio_config,
            video=video_config,
            processing=processing_config
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> "SyncTalkConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            SyncTalkConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "model": asdict(self.model),
            "audio": asdict(self.audio),
            "video": asdict(self.video),
            "processing": asdict(self.processing)
        }
    
    def to_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON file
        """
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Model settings
        if "SYNCTALK_MODEL_NAME" in os.environ:
            self.model.name = os.environ["SYNCTALK_MODEL_NAME"]
        if "SYNCTALK_DEVICE" in os.environ:
            self.model.device = os.environ["SYNCTALK_DEVICE"]
        if "SYNCTALK_AUDIO_MODE" in os.environ:
            self.model.audio_mode = os.environ["SYNCTALK_AUDIO_MODE"]
            
        # Audio settings
        if "SYNCTALK_VAD_THRESHOLD" in os.environ:
            self.audio.vad_threshold = float(os.environ["SYNCTALK_VAD_THRESHOLD"])
            
        # Video settings
        if "SYNCTALK_FPS" in os.environ:
            self.video.fps = int(os.environ["SYNCTALK_FPS"])
            
        # Processing settings
        if "SYNCTALK_LOG_LEVEL" in os.environ:
            self.processing.log_level = os.environ["SYNCTALK_LOG_LEVEL"]


def get_default_config(model_name: str) -> SyncTalkConfig:
    """
    Get default configuration for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Default configuration
    """
    return SyncTalkConfig(model=ModelConfig(name=model_name))


def load_model_config(model_name: str, config_dir: str = "./configs") -> SyncTalkConfig:
    """
    Load model-specific configuration.
    
    First tries to load from model-specific config file, then falls back to default.
    
    Args:
        model_name: Name of the model
        config_dir: Directory containing config files
        
    Returns:
        Model configuration
    """
    config_path = Path(config_dir) / f"{model_name}.json"
    
    if config_path.exists():
        config = SyncTalkConfig.from_json(str(config_path))
    else:
        config = get_default_config(model_name)
    
    # Always update from environment variables
    config.update_from_env()
    
    return config


# Preset configurations for common scenarios
PRESETS = {
    "high_quality": {
        "video": {
            "crf": 18,
            "face_crop_size": 512,
            "face_output_size": 480
        },
        "audio": {
            "batch_size": 32
        }
    },
    "fast": {
        "video": {
            "crf": 23,
            "fps": 20
        },
        "audio": {
            "batch_size": 128,
            "vad_threshold": 0.6
        }
    },
    "low_memory": {
        "model": {
            "device": "cpu"
        },
        "audio": {
            "batch_size": 16
        },
        "processing": {
            "enable_caching": False
        }
    }
}


def apply_preset(config: SyncTalkConfig, preset_name: str) -> None:
    """
    Apply a preset configuration.
    
    Args:
        config: Configuration to modify
        preset_name: Name of the preset to apply
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    
    # Apply preset values
    for section, values in preset.items():
        section_obj = getattr(config, section)
        for key, value in values.items():
            setattr(section_obj, key, value)