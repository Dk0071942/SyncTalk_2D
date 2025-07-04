#!/bin/bash
# Simple script to train all datasets

GPU_ID=${1:-0}  # Default to GPU 0

echo "Training all datasets on GPU $GPU_ID"
echo "===================================="

# Train each person
bash scripts/train_328.sh LS1 $GPU_ID
bash scripts/train_328.sh 250627_CB $GPU_ID
# Add more datasets here as needed
# bash scripts/train_328.sh person3 $GPU_ID
# bash scripts/train_328.sh person4 $GPU_ID

echo "All training completed!"