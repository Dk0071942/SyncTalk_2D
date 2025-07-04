# SyncTalk Package

This is the refactored SyncTalk 2D package providing a clean, modular implementation of lip-sync video generation.

## Package Structure

```
synctalk/
├── __init__.py                  # Package initialization
├── config.py                    # Configuration management
├── processing/                  # Video processors
│   ├── __init__.py
│   ├── base.py                 # Base classes and interfaces
│   ├── video_preprocessor.py   # Video preprocessing
│   ├── standard_processor.py   # Standard mode processor
│   └── core_clips_processor.py # Core clips mode processor
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── vad.py                 # Voice Activity Detection
│   ├── structures.py          # Data structures
│   ├── clips_manager.py       # Clips management
│   ├── syncnet.py            # SyncNet model (256x256)
│   ├── syncnet_328.py        # SyncNet model (328x328)
│   ├── unet.py               # U-Net model (256x256)
│   ├── unet_328.py           # U-Net model (328x328)
│   ├── datasetsss.py         # Dataset loader (256x256)
│   ├── datasetsss_328.py     # Dataset loader (328x328)
│   └── utils.py              # Core utility functions
└── utils/                      # Utilities
    ├── __init__.py
    ├── face_blending.py       # Face blending functions
    └── video_processor.py     # Unified video processing
```

## Quick Start

### Standard Mode
```python
from synctalk.processing import StandardVideoProcessor

processor = StandardVideoProcessor("MODEL_NAME")
processor.load_models()
processor.generate_video(
    audio_path="audio.wav",
    output_path="output.mp4",
    start_frame=0
)
```

### Core Clips Mode
```python
from synctalk.processing import CoreClipsProcessor

processor = CoreClipsProcessor("MODEL_NAME")
processor.generate_video(
    audio_path="audio.wav",
    output_path="output.mp4",
    vad_threshold=0.5,
    min_silence_duration=0.75
)
```

### Custom Video Mode
```python
from synctalk.processing import StandardVideoProcessor

processor = StandardVideoProcessor("MODEL_NAME")
processor.load_models()

# Process custom video first
processor.process_custom_video("template_video.mp4")

# Generate with processed frames
processor.generate_video(
    audio_path="audio.wav",
    output_path="output.mp4"
)
```

### Configuration
```python
from synctalk.config import get_default_config, apply_preset

# Get default configuration
config = get_default_config("MODEL_NAME")

# Apply quality preset
apply_preset(config, "high_quality")  # Options: default, high_quality, fast, low_memory

# Custom configuration
config.video.fps = 30
config.processing.batch_size = 4
```

## Core Components

### Models
- **U-Net Models**: Main generation models supporting 256x256 and 328x328 resolutions
- **SyncNet Models**: Audio-visual synchronization networks for training
- **Dataset Loaders**: PyTorch datasets for training data management

### Processors
- **StandardVideoProcessor**: Uses pre-extracted frames from training data
- **CoreClipsProcessor**: Intelligently selects video clips based on VAD
- **VideoPreprocessor**: Handles video frame extraction and preprocessing

### Utilities
- **VAD (Voice Activity Detection)**: Detects speech/silence segments
- **Face Blending**: Advanced face mask creation and blending
- **Video Processing**: Unified pipeline for video preprocessing

## Key Features

- **Modular Design**: Clear separation between different processing modes
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Configuration System**: Flexible configuration with validation
- **Progress Tracking**: Built-in progress callbacks for all operations
- **Error Handling**: Robust error handling with informative messages
- **Multi-Resolution**: Support for both 256x256 and 328x328 models

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- soundfile
- librosa
- Additional dependencies in requirements.txt

## Usage Examples

### With Progress Tracking
```python
def progress_callback(current, total, message):
    print(f"Progress: {current}/{total} - {message}")

processor = StandardVideoProcessor("MODEL_NAME")
processor.load_models()
processor.generate_video(
    audio_path="audio.wav",
    output_path="output.mp4",
    progress_callback=progress_callback
)
```

### Batch Processing
```python
import os
from synctalk.processing import StandardVideoProcessor

processor = StandardVideoProcessor("MODEL_NAME")
processor.load_models()

audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
for audio in audio_files:
    output = f"output_{os.path.basename(audio)}.mp4"
    processor.generate_video(audio, output)
```

## Development

### Running Tests
```bash
cd tests
python run_tests.py
```

### Adding New Processors
Extend the `BaseVideoProcessor` class:
```python
from synctalk.processing.base import BaseVideoProcessor

class CustomProcessor(BaseVideoProcessor):
    def generate_video(self, audio_path, output_path, **kwargs):
        # Your implementation
        pass
```

## Documentation

- [Architecture Overview](../docs/REFACTORED_ARCHITECTURE.md)
- [Training Workflow](../docs/TRAINING_WORKFLOW.md)
- [API Reference](../docs/API_REFERENCE.md)

## License

This package is part of the SyncTalk 2D project. See the main project README for license information.