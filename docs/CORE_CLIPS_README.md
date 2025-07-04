# Core Clips Feature Documentation

## Overview

The Core Clips feature provides an alternative video generation mode that uses pre-recorded video clips instead of the training dataset. This approach offers:

- **Efficient processing**: No need to load entire training datasets
- **Natural body movements**: Uses real recorded movements
- **Smart clip selection**: Voice Activity Detection (VAD) selects appropriate clips
- **Seamless transitions**: Clips are designed to loop smoothly

## Directory Structure

```
core_clips/
└── {model_name}/           # e.g., LS1, AD2.2
    ├── *talk*.mp4         # Talking clips (with duration in filename)
    └── silence.mp4        # Silence/idle clip

dataset/
└── {model_name}/
    ├── core_clips/        # Preprocessed core clips (generated)
    │   ├── {clip_name}/
    │   │   ├── full_body_img/   # Extracted frames
    │   │   ├── landmarks/       # Detected landmarks
    │   │   └── info.txt        # Clip metadata
    │   └── ...
    ├── full_body_img/     # Original training data
    └── landmarks/
```

## Setup Instructions

### 1. Prepare Core Clips

Place your video clips in `core_clips/{model_name}/`:
- **Talking clips**: Name with duration (e.g., "02,08s talk.mp4" = 2.08 seconds)
- **Silence clips**: Name as "silence.mp4"
- All clips should loop seamlessly (same start/end frame)

### 2. Preprocess Core Clips

Run the preprocessing script to extract frames and landmarks:

```bash
# Process all models
python3 preprocess_core_clips.py

# Process specific model
python3 preprocess_core_clips.py --model LS1

# Verify processing status
python3 preprocess_core_clips.py --verify
```

**Note**: Preprocessing only needs to be done once. The script will skip already processed clips.

### 3. Use in Gradio Interface

1. Start the Gradio app
2. Upload your audio file
3. In Advanced Options, enable "Use Core Clips Mode"
4. Adjust settings:
   - **VAD Threshold**: Speech detection sensitivity (0.1-0.9)
   - **Min Silence Duration**: Minimum pause duration (0.1-2.0s)
   - **Save VAD Visualization**: Optionally save speech/silence plot
5. Generate video

## How It Works

### Voice Activity Detection (VAD)
- Uses Silero VAD (torch-based) to detect speech and silence segments
- Configurable thresholds for different audio types
- Merges short segments to avoid choppy transitions

### Clip Selection
- **Speech segments** → Uses talking clips
- **Silence segments** → Uses silence clips
- Clips are selected based on duration matching
- Longer segments use clip looping

### Video Generation
1. VAD analyzes audio to identify speech/silence regions
2. Edit Decision List (EDL) maps segments to appropriate clips
3. Lip-sync model processes only face region during speech
4. Original body movements from clips are preserved
5. Final video assembled with synchronized audio

## Technical Details

### Core Components

1. **`vad_torch.py`**: Voice Activity Detection module
   - Silero VAD integration
   - Fallback energy-based detection
   - Segment merging and filtering

2. **`core_clips_manager.py`**: Clip management
   - Loads and categorizes clips
   - Handles duration parsing
   - Manages preprocessed data

3. **`core_clips_processor.py`**: Main processing logic
   - Creates EDL from VAD segments
   - Applies selective lip-sync
   - Assembles final video

4. **`preprocess_core_clips.py`**: Preprocessing script
   - Extracts frames at 25fps
   - Detects facial landmarks
   - Creates dataset structure

### File Naming Convention

For talking clips: `{seconds},{centiseconds}s talk.mp4`
- Example: "02,08s talk.mp4" = 2.08 seconds
- Example: "16,22s talk.mp4" = 16.22 seconds

## Advantages

1. **Performance**: Faster processing, especially for long videos
2. **Quality**: Natural body movements from real recordings
3. **Flexibility**: Easy to add new clips or styles
4. **Consistency**: Predictable results with same clips

## Tips

- Use high-quality clips with clear facial features
- Ensure clips loop seamlessly for best results
- Adjust VAD threshold based on audio clarity
- Lower min silence duration for more responsive transitions
- Enable VAD visualization to debug detection issues

## Troubleshooting

### No landmarks detected
- Ensure face is clearly visible in clips
- Check lighting and resolution
- May need to adjust landmark detector parameters

### Choppy transitions
- Increase min silence duration
- Check if clips loop properly
- Verify VAD threshold settings

### Processing errors
- Run preprocessing script first
- Check file permissions
- Verify ffmpeg is installed