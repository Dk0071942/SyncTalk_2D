# Changelog

## [2025-01-04] Major Training Refactoring & Robustness Improvements

### Overview
This release introduces significant improvements to the training pipeline, making it more robust, maintainable, and user-friendly. The core training logic has been refactored into a modular architecture, with automatic state tracking and intelligent checkpoint management.

### New Features

#### 1. **Modular Training Architecture**
- Created new `synctalk.training` module with organized components:
  - `losses.py` - Loss functions (PerceptualLoss, cosine_loss)
  - `state_manager.py` - Training state persistence and recovery
  - `trainer.py` - Main model training logic
  - `syncnet_trainer.py` - SyncNet-specific training
- Moved core training logic from scripts to the package for better organization

#### 2. **Robust State Tracking**
- Implemented `.training_state.json` file for persistent state tracking
- Automatically tracks:
  - Preprocessing completion status
  - SyncNet training progress (epochs, checkpoints)
  - Main model training progress (epochs, checkpoints)
- Training can be interrupted and resumed without losing progress

#### 3. **Intelligent Checkpoint Management**
- Changed from saving every 25 epochs to saving after every epoch
- Automatic cleanup keeps only the 3 most recent checkpoints to save disk space
- State file tracks up to 10 checkpoint paths for history
- Smart recovery finds the most recent available checkpoint

#### 4. **Training Sufficiency Detection**
- Automatically detects when training is sufficient (>90 epochs)
- Skips retraining if sufficient checkpoints exist
- Use `--force` flag to override and retrain anyway
- Still trains to full 100 epochs when starting fresh

#### 5. **Loss History Tracking & Visualization**
- Saves training loss history as both JSON log and PNG graph
- Tracks loss for every epoch with timestamps
- Generates visual plots showing training progress
- Highlights minimum loss achieved and at which epoch

#### 6. **Improved Data Preprocessing**
- Created unified `preprocess_data.py` script
- Added batch preprocessing script for multiple videos
- Better error handling and progress reporting
- Automatic validation of preprocessed data

#### 7. **Data Processing Refactoring**
- Moved data processing utilities from scripts to `synctalk.utils`:
  - `preprocessing_utils.py` - Status checking, validation, metadata handling
  - `batch_processing.py` - Parallel batch processing with progress tracking
- Enhanced batch processing with:
  - Parallel worker support for faster processing
  - JSON configuration for complex batch jobs
  - Progress tracking and detailed result reporting
  - Automatic skip of already processed datasets
- Scripts now act as thin CLI wrappers using package functionality
- Added deprecation warnings to `data_utils/process.py`

#### 8. **Standardized FFmpeg Encoding**
- Created `synctalk.utils.ffmpeg_utils` module for consistent video encoding
- Standardized all video encoding to use:
  - Codec: libx264 with preset slow
  - Quality: CRF 18 (was inconsistent: 10, 20, or unspecified)
  - Pixel format: yuv420p
  - Color space: bt709 with proper color primaries and transfer
  - Optimization: movflags +faststart for streaming
- Benefits:
  - Consistent video quality across all processing paths
  - Optimal file sizes (proper CRF value)
  - Better compatibility and color reproduction
  - Faster streaming startup

#### 9. **Clean Progress Bar Output**
- Created `synctalk.utils.progress` module for standardized progress bars
- Fixed tqdm creating new lines in console output
- All progress bars now use:
  - `leave=False` to clean up after completion
  - Fixed width (`ncols=100`) for consistency
  - Proper position management for nested bars
  - Thread-safe console output with `safe_print()`
- Benefits:
  - Clean, professional console output
  - No more scattered progress bars
  - Easier to trace logs and debug
  - Better user experience

#### 10. **SyncNet Training Continuation**
- Added checkpoint continuation support to SyncNet trainer
- Training can now resume from existing checkpoints instead of starting over
- Features:
  - Automatically detects existing checkpoints and continues from there
  - Loads model state, optimizer state, and training history
  - Preserves best loss value for consistent checkpoint saving
  - Example: If checkpoint 50 exists, training continues from epoch 51
- Benefits:
  - No wasted computation when training is interrupted
  - Seamless continuation of training sessions
  - Better resource utilization

### Changed

#### Scripts Refactoring
- **Removed deprecated scripts:**
  - `scripts/train.py` (replaced by train_328.py)
  - `scripts/inference_328.py` (functionality in inference_cli.py)
  - `scripts/training.sh`, `training_328.sh`, `training_328.bat` (outdated)

- **Updated scripts:**
  - `train_328.py` - Now a thin CLI wrapper using synctalk.training modules
  - `preprocess_data.py` - Unified preprocessing script using synctalk.utils
  - `batch_preprocess.py` - Enhanced with parallel processing and better configuration

#### Core Module Updates
- `synctalk/core/syncnet_328.py` - Removed training code (moved to syncnet_trainer.py)
- Fixed PyTorch deprecation warning: `pretrained=True` → `weights=VGG19_Weights.IMAGENET1K_V1`
- Updated imports and module organization throughout
- Updated all video processing modules to use standardized FFmpeg parameters

#### SyncNet Training
- Fixed checkpoint directory structure to use `./syncnet_ckpt/name/`
- Only saves checkpoints when loss improves (not every epoch)
- Maintains separate loss history for SyncNet training

### Fixed

#### Bugs Resolved
- Fixed SyncNet checkpoint directory saving to wrong location
- Fixed training not saving checkpoints after each epoch
- Fixed deprecation warnings in PyTorch model loading
- Fixed state recovery when only 3 physical checkpoints exist

#### Training Reliability
- Training now properly resumes from interruptions
- Checkpoint management prevents disk space issues
- State tracking ensures no lost progress

### Documentation Updates

- Updated `Claude.md` with new project guidelines and patterns
- Added `PROJECT_DOCS.md` for architecture documentation
- Created `docs/TRAINING_WORKFLOW.md` for training pipeline details
- Enhanced README files with clearer instructions
- Updated architecture documentation to reflect new structure

### Migration Guide

For users upgrading from previous versions:

1. **Training Scripts**: Use `train_328.py` instead of deprecated scripts
2. **Preprocessing**: Use the new `preprocess_data.py` script
3. **Checkpoints**: Old checkpoints remain compatible
4. **State Files**: New `.training_state.json` will be created automatically

### Example Usage

```bash
# Preprocess single video
python scripts/preprocess_data.py --video_path video.mp4 --name my_model

# Batch preprocess multiple videos
python scripts/batch_preprocess.py --video_dir ./videos/ --workers 4

# Create batch configuration template
python scripts/batch_preprocess.py --create_template

# Train with automatic state management
python scripts/train_328.py --name my_model --train_syncnet

# Resume interrupted training
python scripts/train_328.py --name my_model --continue_training

# Force retrain even if sufficient
python scripts/train_328.py --name my_model --force
```

### Technical Details

- **State file location**: `dataset/[name]/.training_state.json`
- **Checkpoint retention**: Physical files limited to 3, state tracks 10
- **Loss graphs**: Saved as `loss_history.png` and `training_log.json`
- **SyncNet directory**: `./syncnet_ckpt/[name]/`
- **Main model directory**: `./checkpoint/[name]/`

### Contributors
- Refactoring and improvements implemented with Claude Code assistance

---

This release significantly improves the robustness and usability of the SyncTalk 2D training pipeline. The modular architecture makes it easier to maintain and extend, while the automatic state management ensures reliable training even with interruptions.