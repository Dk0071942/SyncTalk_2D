# Face Blending Fix for Core Clips

## Issue Description

When using core clips, the reconstructed face may not match the body properly, causing:
- Broken neck/head boundaries
- Visible seams between face and body
- Color mismatches
- Unnatural transitions

## Root Causes

1. **Hard Boundary Replacement**: The original implementation uses rectangular face regions without blending
2. **Landmark Misalignment**: Core clips may have slightly different face poses than training data
3. **Color Differences**: Lighting/color statistics may differ between generated face and original body
4. **Different Face Shapes**: Core clips faces may have different proportions than training data

## Implemented Solutions

### 1. Smooth Face Masking (`face_blending_utils.py`)

Instead of hard rectangular boundaries, we now use:
- Smooth face masks based on facial landmarks
- Gaussian blur for feathered edges
- Adjustable expansion ratio for mask size
- Natural blending between face and body

### 2. Color Matching

- Histogram matching in LAB color space
- Matches generated face colors to original image
- Reduces visible color discontinuities

### 3. Improved Blending Pipeline

The new `_generate_frame_with_blending()` method:
1. Generates lip-synced face using standard method
2. Creates smooth face mask from landmarks
3. Matches colors between generated and original
4. Blends using feathered mask

## Debugging Tools

### Visualize Face Masks
```bash
python test_face_blending.py --model LS1 --clip silence
```

This creates a visualization showing:
- Original image with landmarks
- Face masks with different expansion ratios
- Jaw line and neck region
- Mask overlay on image

### Check Landmark Statistics
```bash
python test_face_blending.py --model LS1 --stats
```

Shows landmark statistics across all clips to identify:
- Face size variations
- Potential alignment issues
- Consistency across clips

## Fine-tuning Parameters

In `core_clips_processor.py`:

```python
# Adjust mask expansion (default: 1.15)
face_mask = create_face_mask(lms, img.shape, expansion_ratio=1.15)

# Adjust feather amount (default: 30)
result = blend_faces(img, matched_generated, lms, feather_amount=30)
```

### Parameter Guidelines

- **expansion_ratio**: 
  - Lower (1.0-1.1): Tighter mask, less blending
  - Higher (1.2-1.3): Larger mask, more blending
  - Default 1.15 works well for most cases

- **feather_amount**:
  - Lower (10-20): Sharper transitions
  - Higher (30-50): Smoother transitions
  - Adjust based on face size

## Additional Considerations

### 1. Landmark Quality
Ensure landmarks are detected correctly:
```bash
python preprocess_core_clips.py --verify
```

### 2. Face Alignment
If faces in core clips are very different from training data:
- Consider using only clips with similar face poses
- Use the `align_landmarks_to_reference()` function
- May need to retrain with more diverse data

### 3. Lighting Consistency
For best results:
- Use core clips with consistent lighting
- Similar background to training data
- Avoid extreme shadows or highlights

## Testing Improvements

1. Generate a short test video:
```bash
# Use a simple audio with clear speech/silence
python app_gradio.py
```

2. Enable core clips mode and generate
3. Check neck/head boundaries in output
4. Adjust parameters if needed

## Future Improvements

1. **Poisson Blending**: More sophisticated seamless cloning
2. **Neural Blending**: Learn optimal blending masks
3. **Temporal Smoothing**: Ensure consistency across frames
4. **Adaptive Parameters**: Automatically adjust based on face size

## Fallback Options

If blending issues persist:
1. Use original `generate_frame()` without blending
2. Use parsing masks if available for the model
3. Limit to clips with best landmark quality
4. Consider retraining with core clips included