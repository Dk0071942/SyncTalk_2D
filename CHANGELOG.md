# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed SyncNet training numerical instability on H100 GPUs by properly normalizing cosine similarity loss
- Fixed SyncNet checkpoint loading to handle both old (state dict) and new (with metadata) formats
- Fixed epoch tracking to read actual epochs from checkpoint filename instead of target epochs
- Fixed progress bar creating multiple lines in terminal during frame extraction
- Made loss plotting robust to handle missing or corrupted data without crashing
- Fixed PyRight type checking issue in losses.py
- Added gradient clipping to prevent training instability
- Added weights_only=True to torch.load calls to eliminate FutureWarning

### Changed
- Removed early stopping from SyncNet training to allow achieving optimal low loss values
- Improved error handling throughout training pipeline