# SyncTalk 2D Core Video Processing Refactoring TODO List

## Overview
This document tracks the progress of refactoring the core video generation components into a clear, modular structure.

## Status Legend
- [ ] Not started
- [x] Completed
- [~] In progress
- [!] Blocked/Issue

## Phase 1: Create New Directory Structure

### Main Package Structure
- [x] Create `synctalk/` package directory
- [x] Create `synctalk/__init__.py`

### Processing Submodule
- [x] Create `synctalk/processing/` directory
- [x] Create `synctalk/processing/__init__.py`

### Core Submodule  
- [x] Create `synctalk/core/` directory
- [x] Create `synctalk/core/__init__.py`

### Utils Submodule
- [x] Create `synctalk/utils/` directory
- [x] Create `synctalk/utils/__init__.py`

## Phase 2: Extract and Relocate Core Components

### 2.1 Video Processing Components
- [x] Create `synctalk/processing/video_preprocessor.py`
- [x] Move `VideoProcessor` class from `inference_module.py`
- [x] Add comprehensive type hints
- [x] Update all docstrings
- [x] Update imports in the new module

### 2.2 Face Blending Utilities
- [x] Create `synctalk/utils/face_blending.py`
- [x] Move functions from `face_blending_utils.py`:
  - [x] `create_face_mask`
  - [x] `get_face_region_with_padding`
  - [x] `blend_faces`
  - [x] `match_color_histogram`
  - [x] `align_landmarks_to_reference`
- [x] Add type hints to all functions
- [ ] Create unit tests for face blending

### 2.3 Voice Activity Detection
- [x] Create `synctalk/core/vad.py`
- [x] Move `SileroVAD` class
- [x] Move `AudioSegment` dataclass
- [x] Preserve test_vad() function
- [x] Add error handling improvements

### 2.4 Data Structures
- [x] Create `synctalk/core/structures.py`
- [x] Move `CoreClip` from `frame_based_structures.py`
- [x] Move `FrameBasedClipSelection` from `frame_based_structures.py`
- [x] Move `EditDecisionItem` from `core_clips_processor.py`
- [x] Add validation methods
- [x] Add __eq__ methods for testing

### 2.5 Core Clips Management
- [x] Create `synctalk/core/clips_manager.py`
- [x] Move `CoreClipsManager` class
- [x] Add error handling for missing clips
- [x] Implement clip metadata caching
- [x] Add clip selection algorithms

## Phase 3: Refactor Main Processors

### 3.1 Standard Video Processor
- [x] Create `synctalk/processing/standard_processor.py`
- [x] Move and rename `SyncTalkInference` → `StandardVideoProcessor`
- [x] Refactor `generate_video` into smaller methods:
  - [x] `_prepare_audio_features()`
  - [x] `_load_frame_data()`
  - [x] `_generate_single_frame()`
  - [x] `_write_video_file()`
- [x] Replace `os.system` with `subprocess.run`
- [x] Add progress callback support
- [x] Implement error recovery

### 3.2 Core Clips Video Processor
- [x] Create `synctalk/processing/core_clips_processor.py`
- [x] Move `CoreClipsProcessor` class
- [x] Refactor main method into:
  - [x] `_analyze_audio_segments()`
  - [x] `_create_edit_decision_list()`
  - [x] `_process_speech_segment()`
  - [x] `_process_silence_segment()` 
  - [x] `_assemble_final_video()`
- [x] Update all imports
- [x] Add comprehensive logging
- [x] Replace os.system with subprocess

## Phase 4: Create Base Classes and Interfaces

### 4.1 Base Video Processor
- [x] Create `synctalk/processing/base.py`
- [x] Define `BaseVideoProcessor` abstract class
- [x] Define common interface:
  - [x] `prepare_audio()`
  - [x] `generate_video()`
  - [x] `cleanup()`
- [x] Implement shared utilities (ProgressTracker)

### 4.2 Configuration Management
- [x] Create `synctalk/config.py`
- [x] Define configuration dataclasses:
  - [x] `ModelConfig`
  - [x] `AudioConfig`
  - [x] `VideoConfig`
  - [x] `ProcessingConfig`
- [x] Add validation logic
- [x] Support environment variables

## Phase 5: Update Integration Points

### 5.1 Update Gradio Application
- [x] Create `app_gradio_refactored.py` with new imports
- [x] Import `StandardVideoProcessor`
- [x] Import `CoreClipsProcessor`
- [x] Add quality presets support
- [x] Improve UI with mode-specific options

### 5.2 Update CLI Scripts
- [x] Create `inference_refactored.py` with new structure
- [x] Create `run_synctalk.py` unified entry point
- [x] Add support for both modes
- [x] Add model info and list commands

## Phase 6: Testing and Documentation

### 6.1 Create Test Suite
- [x] Create `tests/` directory structure
- [x] Unit tests:
  - [x] Test data structure validation
  - [x] Test configuration management
  - [ ] Test VAD functionality (requires audio files)
  - [ ] Test face blending operations (requires OpenCV)
- [x] Create test runner script (`run_tests.py`)
- [ ] Integration tests (requires model files)

### 6.2 Update Documentation
- [x] Create `docs/REFACTORED_ARCHITECTURE.md`
- [x] Add comprehensive module documentation
- [x] Create migration guide section
- [x] Document all improvements and changes
- [x] Update inline code documentation

## Phase 7: Cleanup and Optimization

### 7.1 Remove Deprecated Files
- [x] Verify all tests pass
- [x] Create backup branch
- [x] Remove old files:
  - [x] `inference_module.py`
  - [x] `core_clips_processor.py`
  - [x] `core_clips_manager.py`
  - [x] `vad_torch.py`
  - [x] `face_blending_utils.py`
  - [x] `frame_based_structures.py`

### 7.2 Performance Optimization
- [ ] Profile generation performance
- [ ] Implement frame caching
- [ ] Optimize memory usage
- [ ] Add parallel processing
- [ ] Benchmark improvements

## Phase 8: Future Enhancements (Optional)

### 8.1 Advanced Features
- [ ] Multiple face tracking support
- [ ] Smooth clip transitions
- [ ] Real-time preview
- [ ] Additional codec support

### 8.2 Code Quality
- [ ] Add mypy type checking
- [ ] Setup pre-commit hooks
- [ ] Configure CI/CD
- [ ] Add performance benchmarks

---

## Notes

### Important Considerations
1. **Backward Compatibility**: All existing functionality must be preserved
2. **Testing**: Each phase must be tested before proceeding
3. **Documentation**: Update as you implement, not after
4. **Version Control**: Use feature branch `refactor/core-video-processing`

### Progress Tracking
- Started: 2025-07-03
- Last Updated: 2025-07-03
- Current Phase: 7.1 (Completed)
- Estimated Completion: Core refactoring complete. Phase 7.2 and 8 are optional optimizations

### Issues/Blockers
- None currently

### Decisions Made
- Using `synctalk` as the main package name
- Separating utilities into their own submodule
- Creating base classes for extensibility