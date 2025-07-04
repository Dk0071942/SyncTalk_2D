#!/bin/bash

# This script processes all audio files in a specified directory to generate avatar videos.
#
# Usage:
# ./batch_inference.sh <avatar_name> <audio_directory>
#
# Example from your request:
# ./batch_inference.sh LS1 audio_material/extracted_audio_LS1

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <avatar_name> <audio_directory>"
    exit 1
fi

# Assign arguments to variables
AVATAR_NAME=$1
AUDIO_DIR=$2

# Check if the audio directory exists
if [ ! -d "$AUDIO_DIR" ]; then
  echo "Error: Directory '$AUDIO_DIR' not found."
  exit 1
fi

echo "Starting batch processing for avatar '$AVATAR_NAME' with audio from '$AUDIO_DIR'..."

# Find and process all files in the audio directory.
find "$AUDIO_DIR" -type f -print0 | while IFS= read -r -d $'\0' audio_file; do
  echo "Processing '$audio_file'..."
  python scripts/inference_cli.py --name "$AVATAR_NAME" --audio_path "$audio_file" < /dev/null
done

echo "Batch processing finished." 