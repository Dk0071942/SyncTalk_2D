"""
Preprocessing utilities for SyncTalk 2D.

This module provides utility functions for checking preprocessing status,
validating datasets, and tracking preprocessing progress.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, Optional


def check_preprocessing_status(dataset_path: str, expected_frames: Optional[int] = None) -> Dict[str, any]:
    """
    Check the preprocessing status of a dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        expected_frames: Expected number of frames (optional)
        
    Returns:
        Dictionary with status information including:
        - is_complete: Whether preprocessing is complete
        - status: Status message
        - details: Detailed information about each component
    """
    status = {
        'is_complete': False,
        'status': 'Not started',
        'details': {
            'frames': {'exists': False, 'count': 0},
            'landmarks': {'exists': False, 'count': 0},
            'audio': {'exists': False},
            'audio_features': {'ave': False, 'hubert': False, 'wenet': False}
        }
    }
    
    if not os.path.exists(dataset_path):
        status['status'] = 'Dataset directory does not exist'
        return status
    
    # Check frames
    frames_dir = os.path.join(dataset_path, 'full_body_img')
    if os.path.exists(frames_dir):
        frames = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
        status['details']['frames']['exists'] = True
        status['details']['frames']['count'] = len(frames)
    
    # Check landmarks
    landmarks_dir = os.path.join(dataset_path, 'landmarks')
    if os.path.exists(landmarks_dir):
        landmarks = [f for f in os.listdir(landmarks_dir) if f.endswith('.lms')]
        status['details']['landmarks']['exists'] = True
        status['details']['landmarks']['count'] = len(landmarks)
    
    # Check audio
    audio_path = os.path.join(dataset_path, 'aud.wav')
    status['details']['audio']['exists'] = os.path.exists(audio_path)
    
    # Check audio features
    feature_files = {
        'ave': 'aud_ave.npy',
        'hubert': 'aud_hubert.npy',
        'wenet': 'aud_wenet.npy'
    }
    
    for feature_type, filename in feature_files.items():
        feature_path = os.path.join(dataset_path, filename)
        status['details']['audio_features'][feature_type] = os.path.exists(feature_path)
    
    # Determine overall status
    frames_ok = status['details']['frames']['count'] > 0
    landmarks_ok = status['details']['landmarks']['count'] > 0
    audio_ok = status['details']['audio']['exists']
    any_features_ok = any(status['details']['audio_features'].values())
    
    # Also check if landmarks exist but count is 0 (could be counting issue)
    if not landmarks_ok and os.path.exists(landmarks_dir):
        # Double-check with all files, not just .lms
        all_landmark_files = os.listdir(landmarks_dir)
        if len(all_landmark_files) > 0:
            status['details']['landmarks']['exists'] = True
            status['details']['landmarks']['count'] = len(all_landmark_files)
            landmarks_ok = True
    
    if frames_ok and landmarks_ok and audio_ok and any_features_ok:
        status['is_complete'] = True
        status['status'] = 'Complete'
        
        # Check frame/landmark count match
        if status['details']['frames']['count'] != status['details']['landmarks']['count']:
            status['status'] = 'Warning: Frame/landmark count mismatch'
        elif expected_frames and status['details']['frames']['count'] != expected_frames:
            status['status'] = f'Warning: Expected {expected_frames} frames, found {status["details"]["frames"]["count"]}'
    else:
        missing = []
        if not frames_ok:
            missing.append('frames')
        if not landmarks_ok:
            missing.append('landmarks')
        if not audio_ok:
            missing.append('audio')
        if not any_features_ok:
            missing.append('audio features')
        status['status'] = f'Incomplete - missing: {", ".join(missing)}'
    
    return status


def validate_preprocessed_data(dataset_path: str, asr_mode: str = 'ave') -> Tuple[bool, str]:
    """
    Validate that preprocessed data is ready for training.
    
    Args:
        dataset_path: Path to the dataset directory
        asr_mode: ASR model type to check for
        
    Returns:
        (is_valid, message) tuple
    """
    status = check_preprocessing_status(dataset_path)
    
    if not status['is_complete']:
        return False, f"Preprocessing incomplete: {status['status']}"
    
    # Check specific ASR feature
    feature_file = f'aud_{asr_mode}.npy'
    feature_path = os.path.join(dataset_path, feature_file)
    
    if not os.path.exists(feature_path):
        return False, f"Missing required audio feature file: {feature_file}"
    
    # Validate feature dimensions
    try:
        features = np.load(feature_path)
        if len(features.shape) != 2:
            return False, f"Invalid audio feature shape: expected 2D array, got {features.shape}"
    except Exception as e:
        return False, f"Error loading audio features: {str(e)}"
    
    # Check frame/landmark consistency
    frames_count = status['details']['frames']['count']
    landmarks_count = status['details']['landmarks']['count']
    
    if frames_count != landmarks_count:
        return False, f"Frame/landmark count mismatch: {frames_count} frames, {landmarks_count} landmarks"
    
    return True, f"Dataset validated: {frames_count} frames ready for training"


def get_preprocessing_info(dataset_path: str) -> str:
    """
    Get a human-readable summary of preprocessing status.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Formatted string with preprocessing information
    """
    status = check_preprocessing_status(dataset_path)
    
    info = [f"Dataset: {os.path.basename(dataset_path)}"]
    info.append(f"Status: {status['status']}")
    info.append("")
    
    if status['details']['frames']['exists']:
        info.append(f"Frames: {status['details']['frames']['count']}")
    else:
        info.append("Frames: Not extracted")
    
    if status['details']['landmarks']['exists']:
        info.append(f"Landmarks: {status['details']['landmarks']['count']}")
    else:
        info.append("Landmarks: Not detected")
    
    if status['details']['audio']['exists']:
        info.append("Audio: Extracted")
    else:
        info.append("Audio: Not extracted")
    
    features = []
    for feature_type, exists in status['details']['audio_features'].items():
        if exists:
            features.append(feature_type)
    
    if features:
        info.append(f"Audio features: {', '.join(features)}")
    else:
        info.append("Audio features: None")
    
    return "\n".join(info)


def save_preprocessing_metadata(dataset_path: str, metadata: Dict[str, any]) -> None:
    """
    Save preprocessing metadata to a JSON file.
    
    Args:
        dataset_path: Path to the dataset directory
        metadata: Metadata dictionary to save
    """
    metadata_path = os.path.join(dataset_path, '.preprocessing_metadata.json')
    
    # Add timestamp
    import time
    metadata['timestamp'] = time.time()
    metadata['dataset_path'] = dataset_path
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_preprocessing_metadata(dataset_path: str) -> Optional[Dict[str, any]]:
    """
    Load preprocessing metadata from a JSON file.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = os.path.join(dataset_path, '.preprocessing_metadata.json')
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None