# Core Clips Preprocessing Note

The preprocessing script `preprocess_core_clips.py` requires the following dependencies:
- OpenCV (cv2)
- The project's landmark detection module

To run the preprocessing, you need to:

1. **Activate the appropriate Python environment** that has all SyncTalk 2D dependencies installed
2. **Run the preprocessing script**:
   ```bash
   python preprocess_core_clips.py
   ```

## Manual Preprocessing Instructions

If you cannot run the automated script, the preprocessing creates this structure:

```
dataset/
└── LS1/
    └── core_clips/
        ├── 02,08s talk/
        │   ├── full_body_img/
        │   │   ├── 0.jpg
        │   │   ├── 1.jpg
        │   │   └── ...
        │   ├── landmarks/
        │   │   ├── 0.lms
        │   │   ├── 1.lms
        │   │   └── ...
        │   └── info.txt
        ├── silence/
        │   ├── full_body_img/
        │   ├── landmarks/
        │   └── info.txt
        └── ... (other clips)
```

## What the Preprocessing Does

For each video clip in `core_clips/{model_name}/`:
1. Converts to 25fps if needed
2. Extracts all frames as JPG images
3. Detects facial landmarks for each frame
4. Saves metadata about the clip

## Alternative: Runtime Processing

If preprocessing is not possible, the system will:
1. Process clips on-demand during first use
2. Cache results in `core_clips_cache/` directory
3. Reuse cached data on subsequent runs

This is slower for the first run but still functional.

## Using Core Clips Without Preprocessing

The core clips feature will work even without preprocessing:
- First run will be slower as it processes clips on-demand
- Subsequent runs will use cached data
- You'll see a message: "Processing {clip} on-the-fly (consider running preprocess_core_clips.py)"

To use core clips mode in Gradio:
1. Enable "Use Core Clips Mode" in Advanced Options
2. The system will automatically handle processing as needed