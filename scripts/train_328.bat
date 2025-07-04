@echo off
REM Windows training script for SyncTalk 2D using separated preprocessing and training
REM Usage: training_328.bat file_name cuda_id

if "%~2"=="" (
    echo Error: You must specify both file_name and cuda_id.
    echo Usage: %0 file_name cuda_id
    echo Example: %0 May 0
    exit /b 1
)

set file_name=%1
set cuda_id=%2
set asr=ave
set file_path=./dataset/%file_name%/%file_name%.mp4

echo === Starting training pipeline for %file_name% on GPU %cuda_id% ===

REM Step 1: Preprocess video
echo Step 1: Preprocessing video...
set CUDA_VISIBLE_DEVICES=%cuda_id%
python scripts/preprocess_data.py --video_path %file_path% --name %file_name% --asr_model %asr%
if errorlevel 1 (
    echo Error preprocessing video!
    exit /b 1
)

REM Step 2: Train model (including SyncNet)
echo Step 2: Training model...
python scripts/train_328.py --name %file_name% --train_syncnet --checkpoint_dir ./checkpoint --asr %asr%
if errorlevel 1 (
    echo Error training model!
    exit /b 1
)

echo === Training complete! ===
echo Model saved to ./checkpoint/%file_name%/