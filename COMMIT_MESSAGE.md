# Suggested Commit Message

## Title
refactor: major training pipeline improvements with robust state management

## Body
### Added
- New modular `synctalk.training` package with organized components
- Automatic training state persistence with `.training_state.json`
- Checkpoint saving after every epoch with automatic cleanup (keep 3)
- Training sufficiency detection (>90 epochs considered sufficient)
- Loss history tracking with JSON logs and visual graphs
- Unified preprocessing script with batch processing support
- Data processing utilities moved to `synctalk.utils` (preprocessing_utils, batch_processing)
- Parallel batch video processing with progress tracking
- Preprocessing metadata and validation functions
- Standardized FFmpeg encoding module (`ffmpeg_utils.py`) with consistent parameters
- Clean progress bar module (`progress.py`) to prevent console output issues
- SyncNet checkpoint continuation support (resume from existing checkpoints)

### Changed
- Refactored training logic from scripts into reusable modules
- Scripts now act as thin CLI wrappers using package functionality
- Updated checkpoint management to save space while preserving history
- Fixed PyTorch deprecation warnings (VGG19_Weights)
- Improved error handling and progress reporting
- Added deprecation warnings to `data_utils/process.py`
- Standardized all FFmpeg encoding (CRF 18, preset slow, bt709 color space, yuv420p)
- Fixed tqdm progress bars to use leave=False and consistent formatting

### Fixed
- SyncNet checkpoint directory structure (./syncnet_ckpt/name/)
- Training checkpoint saving frequency (every epoch instead of every 25)
- State recovery when physical checkpoints are limited

### Removed
- Deprecated training scripts (train.py, training.sh variants)
- Old inference_328.py (merged into inference_cli.py)

The training pipeline is now more robust, maintainable, and user-friendly
with automatic state tracking and intelligent resource management.

---

## Alternative Short Version
```
feat: robust training with state persistence & smart checkpoints

- Add training state tracking (.training_state.json)
- Save checkpoints every epoch, keep only 3 recent
- Add loss visualization (graphs + JSON logs)
- Refactor into modular synctalk.training package
- Fix SyncNet checkpoint directory structure
- Remove deprecated scripts
```