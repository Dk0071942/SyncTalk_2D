# SyncTalk 2D Project Documentation

## System Architecture

- **Purpose**: Generate lip-sync videos from audio input using deep learning models
- **Core Approach**: Uses U-Net architecture with MobileNet-inspired components to generate 328x328 resolution videos with accurate lip synchronization
- **Key Technologies**: PyTorch, OpenCV, Gradio, FFmpeg, ONNX Runtime

## Data Flow

[Input Audio/Video] → [Preprocessing] → [Feature Extraction] → [Model Inference] → [Frame Generation] → [Post-processing] → [Output Video]

## Major Components

- **synctalk/**: Core package with modular architecture
  - `core/`: Model implementations (UNet, SyncNet, datasets)
  - `processing/`: Video processors (Standard and Core Clips modes)
  - `utils/`: Utility functions and helpers
  - `config.py`: Configuration management
- **scripts/**: Training and inference scripts
  - `train_328.py`: Main training script for 328x328 models
  - `inference_cli.py`: Command-line interface for inference
  - `preprocess_data.py`: Data preprocessing pipeline
- **app_gradio.py**: Web interface using Gradio
- **run_synctalk.py**: Unified inference interface

## Component Dependencies

```
app_gradio.py → run_synctalk.py → synctalk.processing → synctalk.core
                                                       ↘
scripts/train_328.py → synctalk.core.datasetsss_328 → synctalk.core.unet_328
                    ↘                               ↗
                      synctalk.core.syncnet_328
```

## Entry Points

- **Web Interface**: `app_gradio.py:main()` - Gradio web app for easy interaction
- **CLI Inference**: `scripts/inference_cli.py:main()` - Command-line video generation
- **Training**: `scripts/train_328.py:main()` - Model training pipeline
- **Unified Interface**: `run_synctalk.py:SyncTalkInterface` - Core inference class

## Configuration System

- **Config Files**: 
  - `/configs/`: Configuration directory (if exists)
  - Model checkpoints: `/checkpoint/[model_name]/`
  - Dataset configs: `/dataset/[name]/`
- **Environment Variables**:
  - `CUDA_VISIBLE_DEVICES`: GPU selection
  - Various model parameters configurable via command-line args

## Code Patterns

- **Processor Pattern**: Abstract base class with Standard and CoreClips implementations - Example: `synctalk/processing/base.py:VideoProcessor`
- **Model Loading**: Lazy loading with checkpoint management - Example: `synctalk/core/unet_328.py:Model`
- **Frame Processing**: Batch processing with padding for efficient inference - Example: `synctalk/processing/standard_processor.py:process_frames()`
- **Audio Feature Extraction**: AVE encoder integration - Example: `data_utils/AVE/*.py`

## Common Abstractions

- **VideoProcessor**: Base class for all video processing modes
- **Config**: Centralized configuration management
- **FrameData**: Structured data for frame processing
- **Model**: U-Net based generation model

## Anti-patterns to Avoid

- Don't use `interface{}` or `any{}` in Go code - use concrete types
- Don't use `time.Sleep()` - use channels for synchronization
- Don't keep old and new code together - delete old code when replacing
- Don't create versioned function names (e.g., processV2)

## Quick Find

- **Main model implementation**: `synctalk/core/unet_328.py`
- **Training loop**: `scripts/train_328.py:276` (approx)
- **Inference pipeline**: `synctalk/processing/standard_processor.py:process()`
- **Audio processing**: `data_utils/get_landmark.py:extract_audio_features()`
- **Web interface routes**: `app_gradio.py:process_video()`
- **Configuration defaults**: `synctalk/config.py:get_default_config()`

## Directory Purposes

- `/synctalk/`: Refactored modular package (main codebase)
- `/scripts/`: Standalone scripts for training, inference, preprocessing
- `/data_utils/`: Data preprocessing utilities and audio encoders
- `/model/`: Legacy model files (being phased out)
- `/checkpoint/`: Trained model checkpoints
- `/dataset/`: Training datasets with video/audio/landmarks
- `/demo/`: Demo audio files for testing
- `/docs/`: Project documentation
- `/tests/`: Unit and integration tests
- `/backup_old_code/`: Archive of old implementations
- `/gradio_outputs/`: Generated videos from web interface

## Common Tasks

### Adding a new feature
1. Check `synctalk/processing/base.py` for processor patterns
2. Implement in appropriate processor class
3. Update `synctalk/config.py` if new parameters needed
4. Add tests in `/tests/`

### Training a new model
1. Prepare video: `dataset/[name]/[name].mp4`
2. Run preprocessing: `python scripts/preprocess_data.py`
3. Train model: `python scripts/train_328.py`
4. Model saved to: `checkpoint/[name]/`

### Debugging issues
1. Start at entry point (`app_gradio.py` or `scripts/inference_cli.py`)
2. Check logs in console output
3. Common issues:
   - Module import errors: Check sys.path additions in scripts
   - CUDA/GPU errors: Verify CUDA_VISIBLE_DEVICES
   - Audio sync issues: Check frame alignment in processors
   - Memory issues: Reduce batch_size in training

### Running inference
1. Web: `python app_gradio.py` → http://localhost:7860
2. CLI: `python scripts/inference_cli.py --name [model] --audio_path [audio.wav]`
3. Batch: `./scripts/batch_inference.sh [model] [audio_dir]`

## Recent Changes (as of latest commits)

- Major refactoring into modular `synctalk/` package
- Improved Gradio interface with better error handling
- Fixed training script module imports
- Enhanced documentation structure
- Added CLAUDE.md for development guidelines