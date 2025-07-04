#!/bin/bash
# Dynamic script to train all available datasets

GPU_ID=${1:-0}  # Default to GPU 0
DATASET_DIR="dataset"

echo "Training all datasets on GPU $GPU_ID"
echo "===================================="

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' not found!"
    exit 1
fi

# Find all datasets (directories containing aud.wav file)
echo "Scanning for available datasets..."
datasets=()
for dir in "$DATASET_DIR"/*; do
    if [ -d "$dir" ] && [ -f "$dir/aud.wav" ]; then
        dataset_name=$(basename "$dir")
        datasets+=("$dataset_name")
        echo "  Found: $dataset_name"
    fi
done

# Check if any datasets were found
if [ ${#datasets[@]} -eq 0 ]; then
    echo "No datasets found in $DATASET_DIR"
    echo "Datasets should contain at least an 'aud.wav' file"
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