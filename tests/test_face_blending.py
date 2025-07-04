#!/usr/bin/env python3
"""Test face blending to debug neck/head boundary issues."""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from face_blending_utils import create_face_mask, blend_faces


def visualize_face_mask(model_name: str = "LS1", clip_name: str = "silence"):
    """Visualize the face mask for a sample frame."""
    
    # Load a sample frame and landmarks
    frame_path = Path(f"dataset/{model_name}/core_clips/{clip_name}/full_body_img/0.jpg")
    lms_path = Path(f"dataset/{model_name}/core_clips/{clip_name}/landmarks/0.lms")
    
    if not frame_path.exists() or not lms_path.exists():
        print(f"Sample files not found. Make sure {model_name} is preprocessed.")
        return
        
    # Load image
    img = cv2.imread(str(frame_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load landmarks
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    
    # Create face mask with different expansion ratios
    mask1 = create_face_mask(lms, img.shape, expansion_ratio=1.0)
    mask2 = create_face_mask(lms, img.shape, expansion_ratio=1.15)
    mask3 = create_face_mask(lms, img.shape, expansion_ratio=1.3)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image with landmarks
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    ax.scatter(lms[:, 0], lms[:, 1], c='red', s=10)
    ax.set_title("Original with Landmarks")
    ax.axis('off')
    
    # Masks with different expansions
    for idx, (mask, ratio) in enumerate([(mask1, 1.0), (mask2, 1.15), (mask3, 1.3)]):
        ax = axes[0, idx + 1] if idx < 2 else axes[1, idx - 2]
        ax.imshow(mask, cmap='gray')
        ax.set_title(f"Mask (expansion={ratio})")
        ax.axis('off')
    
    # Overlay masks on image
    for idx, (mask, ratio) in enumerate([(mask2, 1.15)]):
        ax = axes[1, 1]
        overlay = img_rgb.copy()
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(overlay, 0.7, mask_3ch, 0.3, 0)
        ax.imshow(overlay)
        ax.set_title(f"Mask Overlay (expansion={ratio})")
        ax.axis('off')
    
    # Show jaw and neck region
    ax = axes[1, 2]
    neck_region = img_rgb.copy()
    # Draw jaw line
    jaw_points = lms[0:17]
    for i in range(len(jaw_points) - 1):
        cv2.line(neck_region, tuple(jaw_points[i]), tuple(jaw_points[i+1]), (255, 0, 0), 2)
    # Highlight neck area
    neck_y = int(lms[8, 1])  # Bottom of jaw
    cv2.rectangle(neck_region, (0, neck_y), (img.shape[1], neck_y + 50), (0, 255, 0), 2)
    ax.imshow(neck_region)
    ax.set_title("Jaw Line & Neck Region")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("face_mask_visualization.png", dpi=150)
    print("Saved visualization to face_mask_visualization.png")
    plt.show()


def check_landmark_statistics(model_name: str = "LS1"):
    """Check landmark statistics across all clips to identify potential issues."""
    
    core_clips_dir = Path(f"dataset/{model_name}/core_clips")
    
    if not core_clips_dir.exists():
        print(f"No preprocessed clips found for {model_name}")
        return
        
    print(f"\nLandmark Statistics for {model_name}:")
    print("=" * 60)
    
    all_jaw_widths = []
    all_face_heights = []
    
    for clip_dir in core_clips_dir.iterdir():
        if not clip_dir.is_dir():
            continue
            
        lms_dir = clip_dir / "landmarks"
        if not lms_dir.exists():
            continue
            
        clip_jaw_widths = []
        clip_face_heights = []
        
        for lms_file in lms_dir.glob("*.lms"):
            # Load landmarks
            lms_list = []
            with open(lms_file, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            lms = np.array(lms_list)
            
            if len(lms) == 68:  # Standard 68 landmarks
                # Jaw width (from point 0 to 16)
                jaw_width = lms[16, 0] - lms[0, 0]
                # Face height (from chin to eyebrow region)
                face_height = lms[8, 1] - lms[19, 1]  # Chin to left eyebrow
                
                clip_jaw_widths.append(jaw_width)
                clip_face_heights.append(face_height)
                all_jaw_widths.append(jaw_width)
                all_face_heights.append(face_height)
        
        if clip_jaw_widths:
            print(f"\n{clip_dir.name}:")
            print(f"  Jaw width: {np.mean(clip_jaw_widths):.1f} ± {np.std(clip_jaw_widths):.1f}")
            print(f"  Face height: {np.mean(clip_face_heights):.1f} ± {np.std(clip_face_heights):.1f}")
    
    if all_jaw_widths:
        print(f"\nOverall Statistics:")
        print(f"  Jaw width: {np.mean(all_jaw_widths):.1f} ± {np.std(all_jaw_widths):.1f}")
        print(f"  Face height: {np.mean(all_face_heights):.1f} ± {np.std(all_face_heights):.1f}")
        print(f"  Width/Height ratio: {np.mean(all_jaw_widths) / np.mean(all_face_heights):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LS1", help="Model name")
    parser.add_argument("--clip", default="silence", help="Clip name to visualize")
    parser.add_argument("--stats", action="store_true", help="Show landmark statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        check_landmark_statistics(args.model)
    else:
        visualize_face_mask(args.model, args.clip)