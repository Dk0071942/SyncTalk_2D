"""
Face blending utilities for SyncTalk 2D.

This module provides functions for seamlessly blending generated face regions
back into original frames using various masking and color matching techniques.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def create_face_mask(landmarks: np.ndarray, img_shape: Tuple[int, int], 
                     expansion_ratio: float = 1.2) -> np.ndarray:
    """
    Create a smooth mask around the face region using landmarks.
    
    Args:
        landmarks: Facial landmarks array (68 points)
        img_shape: Shape of the image (height, width)
        expansion_ratio: How much to expand the face region
        
    Returns:
        Smooth mask for face region (0-255 grayscale)
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get face boundary points - using jaw line and forehead estimate
    # Jaw line: landmarks 0-16
    # Add some forehead points based on eye positions
    jaw_points = landmarks[0:17]
    
    # Estimate forehead points
    eye_center_y = (landmarks[36:42, 1].mean() + landmarks[42:48, 1].mean()) / 2
    forehead_height = (landmarks[8, 1] - eye_center_y) * 1.5
    
    # Create face contour
    face_contour = []
    
    # Add jaw points
    for i in range(17):
        face_contour.append(jaw_points[i])
    
    # Add forehead points (reverse order)
    forehead_points = []
    for i in [16, 15, 14, 13, 12]:  # Right side up
        x = landmarks[i, 0]
        y = eye_center_y - forehead_height
        forehead_points.append([x, y])
    
    # Add center forehead
    forehead_points.append([landmarks[27, 0], eye_center_y - forehead_height * 1.2])
    
    # Add left side forehead
    for i in [4, 3, 2, 1, 0]:
        x = landmarks[i, 0]
        y = eye_center_y - forehead_height
        forehead_points.append([x, y])
    
    face_contour.extend(forehead_points)
    
    # Convert to numpy array
    face_contour = np.array(face_contour, dtype=np.int32)
    
    # Expand contour slightly
    center = face_contour.mean(axis=0)
    expanded_contour = center + (face_contour - center) * expansion_ratio
    expanded_contour = expanded_contour.astype(np.int32)
    
    # Draw filled polygon
    cv2.fillPoly(mask, [expanded_contour], 255)
    
    # Apply Gaussian blur for smooth edges
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    
    return mask


def get_face_region_with_padding(landmarks: np.ndarray, img_shape: Tuple[int, int],
                                padding_ratio: float = 0.3) -> Tuple[int, int, int, int]:
    """
    Get face region coordinates with padding.
    
    Args:
        landmarks: Facial landmarks (68 points)
        img_shape: Image shape (height, width)
        padding_ratio: How much padding to add (as ratio of face width)
        
    Returns:
        (xmin, ymin, xmax, ymax) with padding applied and bounds checked
    """
    h, w = img_shape[:2]
    
    # Get bounding box from landmarks
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    face_xmin = int(x_coords.min())
    face_xmax = int(x_coords.max())
    face_ymin = int(y_coords.min())
    face_ymax = int(y_coords.max())
    
    # Add padding
    face_width = face_xmax - face_xmin
    face_height = face_ymax - face_ymin
    
    padding_x = int(face_width * padding_ratio)
    padding_y = int(face_height * padding_ratio)
    
    # Apply padding with bounds checking
    xmin = max(0, face_xmin - padding_x)
    xmax = min(w, face_xmax + padding_x)
    ymin = max(0, face_ymin - padding_y)
    ymax = min(h, face_ymax + padding_y)
    
    return xmin, ymin, xmax, ymax


def blend_faces(original_img: np.ndarray, generated_face: np.ndarray,
                landmarks: np.ndarray, feather_amount: int = 20) -> np.ndarray:
    """
    Blend generated face with original image using smooth masking.
    
    Args:
        original_img: Original image (BGR)
        generated_face: Generated face region (same size as original, BGR)
        landmarks: Facial landmarks (68 points)
        feather_amount: Amount of feathering at edges (not used in current implementation)
        
    Returns:
        Blended image (BGR)
    """
    # Create smooth mask
    mask = create_face_mask(landmarks, original_img.shape, expansion_ratio=1.1)
    
    # Normalize mask to 0-1
    mask_norm = mask.astype(np.float32) / 255.0
    
    # Expand to 3 channels
    mask_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
    
    # Blend images
    result = (generated_face * mask_3ch + original_img * (1 - mask_3ch)).astype(np.uint8)
    
    return result


def match_color_histogram(source: np.ndarray, target: np.ndarray, 
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Match color histogram of source to target.
    
    This function adjusts the color distribution of the source image to match
    that of the target image, useful for seamless blending of generated faces.
    
    Args:
        source: Source image to adjust (BGR)
        target: Target image to match (BGR)
        mask: Optional mask for region of interest (0-255 grayscale)
        
    Returns:
        Color-matched source image (BGR)
    """
    # Convert to LAB color space for better color matching
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Get statistics
    if mask is not None:
        mask_bool = mask > 128
        source_mean = source_lab[mask_bool].mean(axis=0)
        source_std = source_lab[mask_bool].std(axis=0)
        target_mean = target_lab[mask_bool].mean(axis=0) 
        target_std = target_lab[mask_bool].std(axis=0)
    else:
        source_mean = source_lab.mean(axis=(0, 1))
        source_std = source_lab.std(axis=(0, 1))
        target_mean = target_lab.mean(axis=(0, 1))
        target_std = target_lab.std(axis=(0, 1))
    
    # Transfer color statistics
    result_lab = (source_lab - source_mean) * (target_std / (source_std + 1e-8)) + target_mean
    
    # Clip values
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)  # L channel
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)  # A channel  
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)  # B channel
    
    # Convert back to BGR
    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return result


def align_landmarks_to_reference(landmarks: np.ndarray, 
                               reference_landmarks: np.ndarray) -> np.ndarray:
    """
    Align landmarks to a reference set using similarity transform.
    
    This helps when core clips have slightly different face poses than training data.
    
    Args:
        landmarks: Current landmarks to align (68 points)
        reference_landmarks: Reference landmarks to align to (68 points)
        
    Returns:
        Aligned landmarks (68 points)
    """
    # Use eye corners and nose tip for alignment (stable points)
    # Left eye: 36, Right eye: 45, Nose tip: 30
    src_points = np.array([
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner  
        landmarks[30]   # Nose tip
    ], dtype=np.float32)
    
    dst_points = np.array([
        reference_landmarks[36],
        reference_landmarks[45],
        reference_landmarks[30]
    ], dtype=np.float32)
    
    # Calculate similarity transform
    M = cv2.getAffineTransform(src_points, dst_points)
    
    # Apply transform to all landmarks
    aligned = np.zeros_like(landmarks, dtype=np.float32)
    for i in range(len(landmarks)):
        pt = np.array([landmarks[i, 0], landmarks[i, 1], 1.0])
        aligned[i] = M.dot(pt)
        
    return aligned.astype(np.int32)