#!/bin/bash
# Dynamic script to train all available datasets
# Usage: ./train_all_simple.sh [GPU_ID] [shutdown]
# Example: ./train_all_simple.sh 0 true  # Train on GPU 0 and shutdown after

GPU_ID=${1:-0}  # Default to GPU 0
SHUTDOWN=${2:-false}  # Default to not shutdown
DATASET_DIR="dataset"

echo "Training all datasets on GPU $GPU_ID"
if [ "$SHUTDOWN" = "true" ]; then
    echo "Server will shutdown after training completes"
fi
echo "===================================="

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' not found!"
    exit 1
fi

# Find all datasets (directories containing at least one mp4 file)
echo "Scanning for available datasets..."
datasets=()
for dir in "$DATASET_DIR"/*; do
    if [ -d "$dir" ] && [ -n "$(find "$dir" -maxdepth 1 -name "*.mp4" -type f | head -1)" ]; then
        dataset_name=$(basename "$dir")
        datasets+=("$dataset_name")
        echo "  Found: $dataset_name"
    fi
done

# Check if any datasets were found
if [ ${#datasets[@]} -eq 0 ]; then
    echo "No datasets found in $DATASET_DIR"
    echo "Datasets should contain at least one '.mp4' file"
    exit 1
fi

echo ""
echo "Found ${#datasets[@]} dataset(s) to train"
echo "===================================="

# Train each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "Training dataset: $dataset"
    echo "------------------------------------"
    bash scripts/train_328.sh "$dataset" "$GPU_ID"
    
    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "Warning: Training failed for dataset $dataset"
    else
        echo "Successfully completed training for $dataset"
    fi
done

echo ""
echo "===================================="
echo "All training completed!"
echo "Trained datasets: ${datasets[*]}"

# Shutdown if requested
if [ "$SHUTDOWN" = "true" ]; then
    echo ""
    echo "Shutting down the server in 100 seconds..."
    sleep 100
    sudo shutdown -h now
fi