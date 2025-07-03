# SyncTalk 2D - Quick Start Guide

## Using the Refactored Code

### 1. Web Interface (Gradio)
```bash
python app_gradio.py
```
Or use the unified entry point:
```bash
python run_synctalk.py web
```

### 2. Command Line Interface

#### Using the unified entry point:
```bash
# Standard mode
python run_synctalk.py generate --name MODEL_NAME --audio audio.wav

# Core clips mode
python run_synctalk.py generate --name MODEL_NAME --audio audio.wav --mode core_clips

# List available models
python run_synctalk.py list

# Show model info
python run_synctalk.py info --name MODEL_NAME
```

#### Using the direct CLI:
```bash
# Standard mode with options
python inference_cli.py --name MODEL_NAME --audio_path audio.wav --start_frame 0

# Core clips mode with VAD options
python inference_cli.py --name MODEL_NAME --audio_path audio.wav --mode core_clips --vad_threshold 0.5

# With quality preset
python inference_cli.py --name MODEL_NAME --audio_path audio.wav --preset high_quality
```

### 3. Python API

```python
# Standard mode
from synctalk.processing import StandardVideoProcessor

processor = StandardVideoProcessor("MODEL_NAME")
processor.load_models()
video_path = processor.generate_video("audio.wav", "output.mp4")

# Core clips mode
from synctalk.processing import CoreClipsProcessor

processor = CoreClipsProcessor("MODEL_NAME")
video_path = processor.generate_video("audio.wav", "output.mp4", vad_threshold=0.5)

# With configuration
from synctalk.config import get_default_config, apply_preset

config = get_default_config("MODEL_NAME")
apply_preset(config, "high_quality")
```

## Quality Presets

- `default`: Balanced quality and speed
- `high_quality`: Higher resolution, better quality (slower)
- `fast`: Optimized for speed (lower quality)
- `low_memory`: For systems with limited resources

## What's New

1. **Cleaner Architecture**: Modular design with clear separation of concerns
2. **Better CLI**: Unified entry point with multiple commands
3. **Configuration System**: Flexible configuration with presets
4. **Improved Gradio UI**: Mode-specific options and quality presets
5. **Type Safety**: Full type hints for better IDE support
6. **Better Error Handling**: More informative error messages

## Migration from Old Code

The refactored code maintains backward compatibility. Old scripts should continue to work, but we recommend updating to use the new structure for better performance and features.

See [REFACTORED_ARCHITECTURE.md](docs/REFACTORED_ARCHITECTURE.md) for detailed documentation.