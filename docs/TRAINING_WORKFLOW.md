# SyncTalk_2D Training Workflow

This document describes the separated data preprocessing and training workflow for SyncTalk_2D.

## Overview

The training process is now split into two distinct phases:
1. **Data Preprocessing** - Extract frames, landmarks, and audio features
2. **Model Training** - Train SyncNet and the main lip-sync model

This separation allows for:
- Preprocessing multiple videos in batch
- Reusing preprocessed data across training runs
- Better debugging and error handling
- More flexible training experiments

## Unified Training Script (Recommended)

The easiest way to train is using the unified training script that handles both preprocessing and training:

### Linux/macOS
```bash
bash scripts/training_328.sh <name> <gpu_id>
```
Example: `bash scripts/training_328.sh May 0`

### Windows
```cmd
scripts\training_328.bat <name> <gpu_id>
```
Example: `scripts\training_328.bat May 0`

This script will:
1. Preprocess the video at `dataset/<name>/<name>.mp4`
2. Train SyncNet for audio-visual synchronization
3. Train the main lip-sync model

## Manual Scripts

### 1. Individual Video Preprocessing
```bash
python scripts/preprocess_data.py \
    --video_path path/to/video.mp4 \
    --name dataset_name \
    --asr_model ave
```

**Options:**
- `--video_path`: Path to input video file (required)
- `--name`: Dataset name for output directory (required)
- `--asr_model`: Audio model [`ave`, `hubert`, `wenet`] (default: ave)
- `--dataset_dir`: Base directory for datasets (default: dataset)
- `--fps`: Target FPS for video (default: 25)
- `--skip_frames`: Skip frame extraction if already done
- `--skip_landmarks`: Skip landmark detection if already done
- `--skip_audio`: Skip audio feature extraction if already done

### 2. Batch Video Preprocessing
```bash
# Process all videos in a directory
python scripts/batch_preprocess.py --video_dir path/to/videos/

# Process videos from a list file
python scripts/batch_preprocess.py --video_list videos.txt

# Process videos from JSON config
python scripts/batch_preprocess.py --json_config config.json
```

**List file format (videos.txt):**
```
/path/to/video1.mp4 person1
/path/to/video2.mp4 person2
# Comments are supported
/path/to/video3.mp4 person3
```

**JSON config format:**
```json
{
  "videos": [
    {"path": "/path/to/video1.mp4", "name": "person1", "asr_model": "ave"},
    {"path": "/path/to/video2.mp4", "name": "person2", "asr_model": "hubert"}
  ]
}
```

### 3. Model Training (After Preprocessing)
```bash
python scripts/train_328.py \
    --name dataset_name \
    --train_syncnet \
    --epochs 100 \
    --batchsize 8 \
    --asr ave
```

**Options:**
- `--name`: Dataset name (must match preprocessed data) - simplified mode
- `--dataset_dir`: Full dataset directory path - compatibility mode
- `--checkpoint_dir`: Base directory for checkpoints (default: checkpoint)
- `--save_dir`: Specific save directory for model checkpoints
- `--batchsize`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--train_syncnet`: Train SyncNet before main model
- `--syncnet_checkpoint_path`: Path to pre-trained SyncNet (skip SyncNet training)
- `--use_syncnet`: Use syncnet loss (requires syncnet_checkpoint)
- `--syncnet_checkpoint`: Direct path to syncnet checkpoint
- `--continue_training`: Continue from latest checkpoint
- `--asr`: ASR model type [`ave`, `hubert`, `wenet`] (default: hubert)
- `--lr`: Learning rate (default: 0.001)
- `--see_res`: Save visualization results during training

**NEW: Automatic State Tracking and Robustness**

The training scripts now include automatic state tracking:

- **State File**: Each dataset has a `.training_state.json` file tracking progress
- **Auto-Resume**: Scripts automatically detect completed steps and skip them
- **Smart SyncNet**: If SyncNet is already trained, it won't retrain
- **Checkpoint Recovery**: Training continues from the last saved checkpoint

**Training State File** (dataset/NAME/.training_state.json):
```json
{
  "preprocessing": {
    "completed": true,
    "frame_count": 1234,
    "landmark_count": 1234
  },
  "syncnet_training": {
    "completed": true,
    "epochs": 100,
    "checkpoint": "./checkpoint/syncnet_NAME/96.pth"
  },
  "main_training": {
    "epochs": 75,
    "checkpoints": ["./checkpoint/NAME/24.pth", "./checkpoint/NAME/49.pth"]
  }
}
```

## Complete Workflow Examples

### Example 1: Single Video Training
```bash
# Step 1: Preprocess the video
python scripts/preprocess_data.py \
    --video_path videos/obama.mp4 \
    --name obama \
    --asr_model ave

# Step 2: Train the model
python scripts/train_328.py \
    --name obama \
    --train_syncnet \
    --epochs 100 \
    --asr ave
```

### Example 2: Multi-Person Training
```bash
# Step 1: Create a video list
cat > videos.txt << EOF
videos/person1.mp4 person1
videos/person2.mp4 person2
videos/person3.mp4 person3
EOF

# Step 2: Batch preprocess all videos
python scripts/batch_preprocess.py \
    --video_list videos.txt \
    --asr_model hubert

# Step 3: Train on one person
python scripts/train_328.py \
    --name person1 \
    --train_syncnet \
    --asr hubert

# Or train on multiple people (requires code modification)
```

### Example 3: Resuming Training
```bash
# Continue training from checkpoint
python scripts/train_328.py \
    --name obama \
    --continue_training \
    --epochs 200 \
    --asr ave  # Must match original training
```

### Example 4: Using Pre-trained SyncNet
```bash
# First person - train with SyncNet
python scripts/train_328.py \
    --name person1 \
    --train_syncnet \
    --asr ave

# Second person - reuse SyncNet
python scripts/train_328.py \
    --name person2 \
    --syncnet_checkpoint_path checkpoint/syncnet_person1/checkpoint_latest.pth \
    --asr ave
```

## Preprocessed Data Structure

After preprocessing, the following structure is created:
```
dataset/
└── {name}/
    ├── full_body_img/      # Extracted video frames
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── landmarks/          # Facial landmarks (PFLD format)
    │   ├── 000000.pkl
    │   ├── 000001.pkl
    │   └── ...
    ├── aud.wav            # Extracted audio (16kHz)
    └── aud_{model}.npy    # Audio features (ave/hubert/wenet)
```

## Tips and Best Practices

1. **Preprocessing Once**: Preprocess your videos once and reuse the data for multiple training runs

2. **Batch Processing**: Use `batch_preprocess.py` when working with multiple videos

3. **ASR Model Selection**:
   - `ave`: Fastest, good for English
   - `hubert`: Better quality, multilingual
   - `wenet`: Chinese language support

4. **Storage Requirements**: Preprocessing requires ~1GB per minute of video (at 328x328 resolution)

5. **Error Recovery**: If preprocessing fails, use `--skip_*` flags to resume from where it stopped

6. **SyncNet Reuse**: Train SyncNet once and reuse for similar speakers/conditions

## Troubleshooting

**"No module named 'synctalk'"**: The training scripts now include proper path setup. If you still see this error, ensure you're running from the project root directory.

**"No audio features found"**: Ensure preprocessing completed successfully and the correct ASR model file exists (e.g., `aud_ave.npy` for ave model)

**"Missing or empty directory"**: Check that preprocessing created all required directories with files

**Out of memory during training**: Reduce `--batchsize` parameter (default is 8, try 4 or 2)

**Preprocessing hangs**: Check that the video file is valid and contains audio

**SyncNet arguments error**: The script now uses the correct arguments. Ensure you're using the latest version.

**Windows path issues**: Use forward slashes (/) or escaped backslashes (\\\\) in paths

**SyncNet training restarts**: Fixed in latest version. The script now checks for existing SyncNet checkpoints before retraining.

**Checkpoint directory issues**: SyncNet checkpoints are now saved in both the main directory and a `syncnet_ckpt` subdirectory for compatibility.