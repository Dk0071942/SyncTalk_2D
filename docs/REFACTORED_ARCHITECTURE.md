# SyncTalk 2D Refactored Architecture

## Overview

This document describes the refactored architecture of SyncTalk 2D, which provides a cleaner, more modular structure while maintaining all existing functionality.

## Package Structure

```
synctalk/
├── __init__.py              # Main package exports
├── README.md                # Package documentation
├── config.py                # Configuration management
├── processing/              # Video processing modules
│   ├── __init__.py
│   ├── base.py             # Base classes and interfaces
│   ├── video_preprocessor.py    # Video preprocessing (frame extraction)
│   ├── standard_processor.py    # Standard mode processor
│   └── core_clips_processor.py  # Core clips mode processor
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── vad.py              # Voice Activity Detection
│   ├── structures.py       # Data structures
│   ├── clips_manager.py    # Core clips management
│   ├── syncnet.py          # SyncNet model (256x256)
│   ├── syncnet_328.py      # SyncNet model (328x328)
│   ├── unet.py             # U-Net model (256x256)
│   ├── unet_328.py         # U-Net model (328x328)
│   ├── datasetsss.py       # Dataset loader (256x256)
│   ├── datasetsss_328.py   # Dataset loader (328x328)
│   └── utils.py            # Core utility functions
└── utils/                   # Utility functions
    ├── __init__.py
    ├── face_blending.py    # Face blending utilities
    └── video_processor.py  # Unified video processing utilities
```

## Key Components

### 1. Processing Modules

#### BaseVideoProcessor (`processing/base.py`)
- Abstract base class defining the common interface
- Provides shared functionality for all processors
- Includes `ProgressTracker` utility class

#### VideoProcessor (`processing/video_preprocessor.py`)
- Handles video preprocessing tasks
- Extracts frames and detects facial landmarks
- Supports custom video templates

#### StandardVideoProcessor (`processing/standard_processor.py`)
- Implements standard mode video generation
- Uses pre-extracted frames from training dataset
- Refactored from `SyncTalkInference` with cleaner structure

#### CoreClipsProcessor (`processing/core_clips_processor.py`)
- Implements core clips mode with VAD
- Dynamically selects pre-recorded video segments
- Clean separation of concerns with dedicated methods

### 2. Core Components

#### VAD Module (`core/vad.py`)
- `SileroVAD`: Voice Activity Detection implementation
- `AudioSegment`: Data structure for audio segments
- Supports both Silero model and energy-based fallback

#### Data Structures (`core/structures.py`)
- `CoreClip`: Represents a video clip with metadata
- `EditDecisionItem`: Edit decision for video assembly
- `FrameBasedClipSelection`: Frame-based clip selection
- All structures include validation methods

#### Clips Manager (`core/clips_manager.py`)
- Manages library of pre-recorded video clips
- Intelligent clip selection algorithms
- Metadata caching for improved performance

#### Model Components
- **SyncNet Models** (`syncnet.py`, `syncnet_328.py`)
  - Audio-visual synchronization network
  - Evaluates lip-sync quality during training
  - Supports both 256x256 and 328x328 resolutions
  
- **U-Net Models** (`unet.py`, `unet_328.py`)
  - Main generation model using U-Net architecture
  - MobileNet-inspired encoder/decoder blocks
  - Audio feature integration for lip-sync
  - Supports 256x256 and 328x328 resolutions

#### Dataset Loaders (`datasetsss.py`, `datasetsss_328.py`)
- Custom PyTorch dataset classes
- Handle frame loading, landmark processing
- Audio feature extraction and alignment
- Support for multiple audio encoders (AVE, Hubert, WaveNet)

### 3. Configuration System

#### Configuration Management (`config.py`)
- Dataclass-based configuration with validation
- Support for JSON serialization/deserialization
- Environment variable support
- Quality presets (high_quality, fast, low_memory)

Configuration hierarchy:
```
SyncTalkConfig
├── ModelConfig       # Model-specific settings
├── AudioConfig       # Audio processing settings
├── VideoConfig       # Video processing settings
└── ProcessingConfig  # Processing behavior settings
```

### 4. Utilities

#### Face Blending (`utils/face_blending.py`)
- Face mask creation and blending functions
- Color histogram matching
- Landmark alignment utilities
- Improved documentation and type hints

#### Video Processing (`utils/video_processor.py`)
- Unified video processing pipeline
- Frame extraction and preprocessing
- Landmark detection integration
- Audio feature extraction coordination

## Processing Workflows

### Standard Mode Workflow

1. **Audio Processing**
   - Load audio file
   - Extract mel-spectrograms
   - Generate audio features using AudioEncoder

2. **Frame Generation**
   - Load frames and landmarks from dataset
   - For each audio frame:
     - Extract face region using landmarks
     - Create masked input (black mouth region)
     - Pass through U-Net with audio features
     - Blend generated face back into frame

3. **Video Assembly**
   - Write frames to temporary video
   - Merge with audio using ffmpeg

### Core Clips Mode Workflow

1. **Audio Analysis**
   - Apply VAD to detect speech/silence segments
   - Merge short segments to avoid choppy transitions

2. **Edit Decision List (EDL)**
   - Map audio segments to appropriate video clips
   - Select "talk" clips for speech, "silence" clips for silence
   - Calculate padding frames if needed

3. **Segment Processing**
   - For speech segments: Apply lip-sync using U-Net
   - For silence segments: Use clips as-is
   - Handle frame padding by repeating last frame

4. **Final Assembly**
   - Concatenate all processed segments
   - Add original audio track

## Integration Points

### CLI Interface

#### Unified Entry Point (`run_synctalk.py`)
```bash
# Generate video
python run_synctalk.py generate --name MODEL --audio audio.wav --mode standard

# Launch web interface
python run_synctalk.py web --port 7860

# Show model info
python run_synctalk.py info --name MODEL

# List available models
python run_synctalk.py list
```

#### Direct Inference (`scripts/inference_cli.py`)
```bash
python scripts/inference_cli.py \
    --name MODEL \
    --audio_path audio.wav \
    --mode core_clips \
    --vad_threshold 0.5
```

### Web Interface

#### Gradio Application (`app_gradio.py`)
- Modern UI with mode-specific options
- Quality preset support
- Real-time progress tracking
- VAD visualization for core clips mode

## Key Improvements

### 1. Modularity
- Clear separation of concerns
- Each module has a single responsibility
- Easy to extend with new processing modes

### 2. Type Safety
- Comprehensive type hints throughout
- Dataclass-based structures with validation
- Better IDE support and error detection

### 3. Error Handling
- Proper exception handling at all levels
- Informative error messages
- Graceful degradation (e.g., VAD fallback)

### 4. Performance
- Metadata caching for core clips
- Subprocess.run instead of os.system
- Progress tracking without blocking

### 5. Configuration
- Centralized configuration management
- Support for different quality presets
- Environment variable overrides

### 6. Testing
- Unit tests for core components
- Test-driven development friendly
- Easy to mock dependencies

## Migration Guide

### For Users

1. **CLI Usage**: Use `run_synctalk.py` or `scripts/inference_cli.py`
2. **Web Interface**: Run `app_gradio.py`
3. **API Changes**: The refactored code maintains backward compatibility

### For Developers

1. **Import Changes**:
   ```python
   # Old
   from inference_module import SyncTalkInference
   from core_clips_processor import CoreClipsProcessor
   
   # New
   from synctalk.processing import StandardVideoProcessor, CoreClipsProcessor
   ```

2. **Configuration**:
   ```python
   # Use configuration system
   from synctalk.config import get_default_config, apply_preset
   
   config = get_default_config("MODEL_NAME")
   apply_preset(config, "high_quality")
   ```

3. **Base Classes**:
   ```python
   # Extend base classes for new processors
   from synctalk.processing.base import BaseVideoProcessor
   
   class CustomProcessor(BaseVideoProcessor):
       # Implement required methods
   ```

## Future Enhancements

1. **Plugin System**: Dynamic loading of custom processors
2. **Distributed Processing**: Support for multi-GPU setups
3. **Real-time Mode**: Streaming video generation
4. **Additional Formats**: Support for more audio/video codecs
5. **Enhanced Blending**: Neural blending techniques