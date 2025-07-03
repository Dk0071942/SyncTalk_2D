# Video-Audio Synchronization Fix Documentation

## Problem Description

The original implementation had synchronization issues when processing video segments:

1. **Partial clip looping**: When a 5-second audio segment required video but only a 4-second clip was available, the system would play 4 seconds, then loop back to play the first 1 second again, causing visual discontinuity.

2. **Time-based calculations**: Using floating-point time calculations and converting to frames caused rounding errors that accumulated over time.

3. **Audio-video mismatch**: Audio features were extracted based on timeline position, but video frames would loop back, causing lip-sync to be out of alignment.

## Solution Overview

The fix implements a frame-based approach that:

1. **Plays full clips first**: When a 5s segment needs a 4s clip, it plays all 100 frames (at 25fps) sequentially.

2. **Pads with last frame**: The remaining 25 frames (1 second) are filled by repeating the last frame of the clip.

3. **Frame-based calculations**: All timing is done in frame numbers to avoid floating-point errors.

## Key Changes

### 1. `CoreClipsManager.select_clips_for_segment()`

**Before:**
```python
def select_clips_for_segment(self, duration: float, clip_type: str) -> List[Tuple[CoreClip, float, float]]:
    # Returns time-based selections that could be partial clips
    # Example: [(clip, 0.0, 4.0), (clip, 0.0, 1.0)]
```

**After:**
```python
def select_clips_for_segment(self, duration: float, clip_type: str, fps: int = 25) -> List[Tuple[CoreClip, int, int, int]]:
    # Returns frame-based selections with padding info
    # Example: [(clip, 0, 100, 25)]  # start_frame, end_frame, padding_frames
```

### 2. `EditDecisionItem` Class

Added frame-based fields:
- `clip_start_frame`: Starting frame number in the clip
- `clip_end_frame`: Ending frame number in the clip  
- `padding_frames`: Number of times to repeat the last frame
- `total_frames`: Property that calculates total output frames

### 3. `CoreClipsProcessor._process_edl_item()`

**Key logic change:**
```python
for i in range(total_frames_needed):
    if i < clip_frames:
        # Play the clip normally
        frame_idx = start_frame + i
    else:
        # Pad with the last frame
        frame_idx = last_frame_idx
```

## Example: 5s Segment with 4s Clip

At 25 fps:
- 5s segment = 125 frames needed
- 4s clip = 100 frames available

Frame sequence:
- Frames 0-99: Play clip frames 0-99 in sequence
- Frames 100-124: Repeat clip frame 99 (last frame)

Audio features:
- Each output frame gets the corresponding audio feature based on its position in the timeline
- This ensures lip-sync remains accurate throughout

## Benefits

1. **Visual continuity**: No jarring jumps when clips loop
2. **Perfect synchronization**: Frame-accurate alignment with audio
3. **No accumulating errors**: Integer frame numbers eliminate rounding issues
4. **Natural appearance**: Padding with the last frame creates a natural "hold" effect

## Testing

Run `test_frame_logic.py` to verify the frame calculations and sequencing logic.