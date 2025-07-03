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
- [ ] Create `synctalk/processing/base.py`
- [ ] Define `BaseVideoProcessor` abstract class
- [ ] Define common interface:
  - [ ] `prepare_audio()`
  - [ ] `generate_video()`
  - [ ] `cleanup()`
- [ ] Implement shared utilities

### 4.2 Configuration Management
- [ ] Create `synctalk/config.py`
- [ ] Define configuration dataclasses:
  - [ ] `ModelConfig`
  - [ ] `AudioConfig`
  - [ ] `VideoConfig`
  - [ ] `ProcessingConfig`
- [ ] Add validation logic
- [ ] Support environment variables

## Phase 5: Update Integration Points

### 5.1 Update Gradio Application
- [ ] Update imports in `app_gradio.py`
- [ ] Import `StandardVideoProcessor`
- [ ] Import `CoreClipsProcessor`
- [ ] Test all modes work correctly
- [ ] Update error messages

### 5.2 Update CLI Scripts
- [ ] Update `inference.py`
- [ ] Update `inference_328.py`
- [ ] Ensure backward compatibility
- [ ] Add deprecation warnings if needed

## Phase 6: Testing and Documentation

### 6.1 Create Test Suite
- [ ] Create `tests/` directory structure
- [ ] Unit tests:
  - [ ] Test VAD functionality
  - [ ] Test face blending operations
  - [ ] Test clip selection logic
  - [ ] Test data structure validation
- [ ] Integration tests:
  - [ ] Test standard video generation
  - [ ] Test core clips generation
  - [ ] Test mode switching
  - [ ] Test error handling

### 6.2 Update Documentation
- [ ] Update `docs/ARCHITECTURE.md`
- [ ] Create `synctalk/README.md`
- [ ] Add module-level documentation
- [ ] Create migration guide
- [ ] Update inline code documentation

## Phase 7: Cleanup and Optimization

### 7.1 Remove Deprecated Files
- [ ] Verify all tests pass
- [ ] Create backup branch
- [ ] Remove old files:
  - [ ] `inference_module.py`
  - [ ] `core_clips_processor.py`
  - [ ] `core_clips_manager.py`
  - [ ] `vad_torch.py`
  - [ ] `face_blending_utils.py`
  - [ ] `frame_based_structures.py`

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
- Current Phase: 3 (Completed)
- Estimated Completion: TBD

### Issues/Blockers
- None currently

### Decisions Made
- Using `synctalk` as the main package name
- Separating utilities into their own submodule
- Creating base classes for extensibility