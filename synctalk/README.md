# SyncTalk Package

This is the refactored SyncTalk 2D package providing a clean, modular implementation of lip-sync video generation.

## Package Structure

```
synctalk/
├── config.py                    # Configuration management
├── processing/                  # Video processors
│   ├── base.py                 # Base classes
│   ├── video_preprocessor.py   # Video preprocessing
│   ├── standard_processor.py   # Standard mode
│   └── core_clips_processor.py # Core clips mode
├── core/                       # Core functionality
│   ├── vad.py                 # Voice Activity Detection
│   ├── structures.py          # Data structures
│   └── clips_manager.py       # Clips management
└── utils/                      # Utilities
    └── face_blending.py       # Face blending functions
```

## Quick Start

### Standard Mode
```python
from synctalk.processing import StandardVideoProcessor

processor = StandardVideoProcessor("MODEL_NAME")
processor.load_models()
processor.generate_video(
    audio_path="audio.wav",
    output_path="output.mp4"
)
```

### Core Clips Mode
```python
from synctalk.processing import CoreClipsProcessor

processor = CoreClipsProcessor("MODEL_NAME")
processor.generate_video(
    audio_path="audio.wav",
    output_path="output.mp4",
    vad_threshold=0.5
)
```

### Configuration
```python
from synctalk.config import get_default_config, apply_preset

config = get_default_config("MODEL_NAME")
apply_preset(config, "high_quality")
```

## Key Features

- **Modular Design**: Clear separation between standard and core clips modes
- **Type Safety**: Comprehensive type hints throughout
- **Configuration**: Flexible configuration with presets
- **Progress Tracking**: Built-in progress callbacks
- **Error Handling**: Robust error handling with informative messages

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Additional dependencies in requirements.txt

## Development

Run tests:
```bash
python run_tests.py
```

For more information, see the [architecture documentation](../docs/REFACTORED_ARCHITECTURE.md).