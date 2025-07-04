# SyncTalk 2D Video Generation Methods

SyncTalk 2D offers three distinct methods for generating lip-synced videos, each optimized for different use cases and requirements.

## Overview

| Method | Source Material | Best For | Key Features |
|--------|----------------|----------|--------------|
| **Training Data Mode** | Pre-extracted dataset frames | Quick generation with consistent quality | Fast processing, predictable results |
| **Custom Video Mode** | User-uploaded video | Matching specific person/style | Flexible, preserves original appearance |
| **Core Clips Mode** | Pre-recorded video segments | Natural motion with speech detection | VAD-based selection, realistic pauses |

## 1. Training Data Mode (Default)

### Description
Uses frames extracted from the original training video stored in the model's dataset directory. This is the fastest and most reliable method.

### How It Works
1. Loads pre-extracted frames from `dataset/{model_name}/full_body_img/`
2. Applies lip-sync predictions to modify mouth region
3. Sequences frames according to audio timing

### Parameters
- **Start Frame**: Which frame to begin from (0-indexed)
- **Loop Back**: Whether to reverse direction at sequence midpoint
- **Audio Encoder**: Choose between ave (default), hubert, or wenet

### Use Cases
- Quick prototyping and testing
- Consistent character appearance
- When original training video style is desired

### Example
```python
# Generate using training data starting from frame 100
processor = StandardVideoProcessor("LS1")
processor.generate_video(
    audio_path="speech.wav",
    start_frame=100,
    loop_back=True
)
```

## 2. Custom Video Mode

### Description
Processes a user-provided video to extract frames and landmarks, then applies lip-sync to match the uploaded video's appearance.

### How It Works
1. Extracts frames from uploaded video
2. Detects facial landmarks for each frame
3. Converts to 25fps if needed
4. Applies lip-sync using extracted frames as source

### Parameters
- **Video Upload**: The source video file
- **All parameters from Training Data Mode** (except start_frame)

### Requirements
- Video must contain clear frontal face views
- Recommended: stable camera, good lighting
- Supported formats: MP4, AVI, MOV

### Use Cases
- Matching a specific person's appearance
- Corporate videos with brand consistency
- Custom character animations

### Example Workflow
1. Upload a video of the target person
2. System extracts and processes frames
3. Lip-sync is applied maintaining original appearance

## 3. Core Clips Mode (VAD-Based)

### Description
Uses Voice Activity Detection (VAD) to intelligently select from pre-recorded video clips, preserving natural motion during speech and silence.

### How It Works
1. Analyzes audio using VAD to detect speech/silence segments
2. Selects appropriate clips:
   - Speech segments: Apply lip-sync
   - Silence segments: Use natural idle/pause clips
3. Seamlessly blends segments

### Parameters
- **VAD Threshold**: Speech detection sensitivity (0.1-0.9)
  - Lower = more sensitive (detects quieter speech)
  - Higher = less sensitive (only clear speech)
- **Minimum Silence Duration**: Shortest pause to preserve (seconds)
- **Visualize VAD**: Generate plot showing speech/silence detection

### Pre-requisites
- Core clips must be pre-processed using `preprocess_core_clips.py`
- Clips stored in `dataset/{model_name}/core_clips/`

### Use Cases
- Natural conversation videos
- Presentations with pauses
- Content requiring realistic idle behavior

### Example
```python
# Generate with intelligent clip selection
processor = CoreClipsProcessor("LS1")
processor.generate_video(
    audio_path="presentation.wav",
    vad_threshold=0.5,
    min_silence_duration=0.75,
    visualize_vad=True
)
```

## Method Comparison

### Performance

| Metric | Training Data | Custom Video | Core Clips |
|--------|---------------|--------------|------------|
| Speed | Fastest | Moderate | Fast |
| Memory Usage | Low | High | Moderate |
| Quality | Consistent | Variable | High |
| Flexibility | Limited | High | Moderate |

### Decision Guide

Choose **Training Data Mode** when:
- You need quick results
- The original training video style is acceptable
- You're doing batch processing

Choose **Custom Video Mode** when:
- You need to match a specific person
- Brand consistency is important
- You have high-quality source video

Choose **Core Clips Mode** when:
- Natural pauses and breathing are important
- The audio has significant silence periods
- You want the most realistic results

## Technical Details

### Frame Processing Pipeline
1. **Face Detection**: SCRFD model for robust face detection
2. **Landmark Detection**: 106-point facial landmarks
3. **Audio Feature Extraction**: Multiple encoder options
4. **Lip-Sync Prediction**: UNet-based architecture
5. **Frame Blending**: Smooth transitions between predictions

### Quality Presets
All modes support quality presets that adjust:
- Resolution scaling
- Batch processing size
- Frame blending parameters
- Memory optimization settings

## Troubleshooting

### Common Issues

1. **"No face detected"**
   - Ensure clear, frontal face view
   - Check lighting conditions
   - Verify video resolution (min 256x256)

2. **"Out of memory"**
   - Use "low_memory" quality preset
   - Process shorter videos
   - Reduce batch size

3. **"Lip-sync not matching"**
   - Check audio quality
   - Verify correct audio encoder selection
   - Ensure audio/video synchronization

### Best Practices

1. **Audio Preparation**
   - Use clean, noise-free audio
   - Normalize audio levels
   - Remove long silences for Training Data mode

2. **Video Requirements**
   - Stable camera position
   - Good lighting on face
   - Minimal occlusions

3. **Performance Optimization**
   - Pre-process videos to 25fps
   - Use appropriate quality presets
   - Batch process multiple files