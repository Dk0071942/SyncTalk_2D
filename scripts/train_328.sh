#!/bin/bash
set -e
# Training script for SyncTalk 2D using separated preprocessing and training
# Usage: bash training_328.sh file_name cuda_id

file_name=$1
cuda_id=$2

if [ -z "$cuda_id" ]; then
    echo "Error: You must specify a cuda_id."
    echo "Usage: $0 file_name cuda_id"
    exit 1
fi

asr="ave"
file_path="./dataset/$file_name/$file_name.mp4"

echo "=== Starting training pipeline for $file_name on GPU $cuda_id ==="

# Step 1: Preprocess video
echo "Step 1: Preprocessing video..."
CUDA_VISIBLE_DEVICES=$cuda_id python scripts/preprocess_data.py \
    --video_path $file_path \
    --name $file_name \
    --asr_model $asr

if [ $? -ne 0 ]; then
    echo "Error during preprocessing!"
    exit 1
fi

# Step 2: Train model (including SyncNet)
echo "Step 2: Training model..."
CUDA_VISIBLE_DEVICES=$cuda_id python scripts/train_328.py \
    --name $file_name \
    --train_syncnet \
    --checkpoint_dir ./checkpoint \
    --asr $asr

if [ $? -ne 0 ]; then
    echo "Error during training!"
    exit 1
fi

echo "=== Training complete! ==="
echo "Model saved to ./checkpoint/$file_name/"
