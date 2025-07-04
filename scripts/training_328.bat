@echo off
REM Windows training script for SyncTalk 2D
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
set data_dir=./dataset/%file_name%

echo Starting training for %file_name% on GPU %cuda_id%...

REM Step 1: Process video
echo Step 1: Processing video...
set CUDA_VISIBLE_DEVICES=%cuda_id%
python data_utils/process.py %file_path%
if errorlevel 1 (
    echo Error processing video!
    exit /b 1
)

REM Step 2: Train SyncNet
echo Step 2: Training SyncNet...
python synctalk/core/syncnet_328.py --save_dir ./syncnet_ckpt/%file_name% --dataset_dir %data_dir% --asr %asr%
if errorlevel 1 (
    echo Error training SyncNet!
    exit /b 1
)

REM Step 3: Find latest checkpoint
echo Step 3: Finding latest checkpoint...
for /f "delims=" %%i in ('dir /b /od "./syncnet_ckpt/%file_name%/*.pth" 2^>nul') do set syncnet_checkpoint=./syncnet_ckpt/%file_name%/%%i
if "%syncnet_checkpoint%"=="" (
    echo Error: No checkpoint found in ./syncnet_ckpt/%file_name%/
    exit /b 1
)
echo Found checkpoint: %syncnet_checkpoint%

REM Step 4: Train main model
echo Step 4: Training main model...
python scripts/train_328.py --dataset_dir %data_dir% --save_dir ./checkpoint/%file_name% --asr %asr% --use_syncnet --syncnet_checkpoint %syncnet_checkpoint%
if errorlevel 1 (
    echo Error training model!
    exit /b 1
)

echo Training complete! Model saved to ./checkpoint/%file_name%/