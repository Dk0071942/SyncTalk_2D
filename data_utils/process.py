import os
import sys
import argparse
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synctalk.utils.video_processor import UnifiedVideoProcessor

# Deprecation warning message
DEPRECATION_MSG = (
    "This function is deprecated and will be removed in a future version. "
    "Please use synctalk.utils.video_processor.UnifiedVideoProcessor instead."
)

# Keep individual functions for backward compatibility
def extract_audio(path, out_path, sample_rate=16000):
    """Extract audio from video (backward compatibility wrapper)."""
    warnings.warn(f"extract_audio: {DEPRECATION_MSG}", DeprecationWarning, stacklevel=2)
    processor = UnifiedVideoProcessor()
    return processor.extract_audio(path, out_path, sample_rate)
    
def extract_images(path, dataset_dir=None):
    """
    Extract images from video (backward compatibility wrapper).
    
    Args:
        path: Video file path
        dataset_dir: Optional dataset directory. If not provided, uses video directory.
    """
    warnings.warn(f"extract_images: {DEPRECATION_MSG}", DeprecationWarning, stacklevel=2)
    if dataset_dir is None:
        # For backward compatibility, use the video's directory
        dataset_dir = os.path.dirname(path)
    
    full_body_dir = os.path.join(dataset_dir, "full_body_img")
    processor = UnifiedVideoProcessor()
    processor.extract_frames(path, full_body_dir, convert_to_25fps=True)
    
def get_audio_feature(wav_path):
    """Extract audio features (backward compatibility wrapper)."""
    warnings.warn(f"get_audio_feature: {DEPRECATION_MSG}", DeprecationWarning, stacklevel=2)
    print("extracting audio feature...")
    os.system("python ./data_utils/ave/test_w2l_audio.py --wav_path "+wav_path)
    
def get_landmark(path, landmarks_dir, dataset_dir=None):
    """
    Detect landmarks (backward compatibility wrapper).
    
    Args:
        path: Video file path (for compatibility)
        landmarks_dir: Directory to save landmarks
        dataset_dir: Optional dataset directory containing full_body_img
    """
    warnings.warn(f"get_landmark: {DEPRECATION_MSG}", DeprecationWarning, stacklevel=2)
    if dataset_dir is None:
        # For backward compatibility
        dataset_dir = os.path.dirname(path)
    
    full_img_dir = os.path.join(dataset_dir, "full_body_img")
    processor = UnifiedVideoProcessor()
    processor.detect_landmarks(full_img_dir, landmarks_dir)

if __name__ == "__main__":
    # Add deprecation warning for the entire script
    warnings.warn(
        "This script (data_utils/process.py) is deprecated. "
        "Please use scripts/preprocess_data.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--dataset_dir', type=str, help="dataset directory (default: video directory)")
    opt = parser.parse_args()

    # Determine dataset directory
    if opt.dataset_dir:
        base_dir = opt.dataset_dir
    else:
        base_dir = os.path.dirname(opt.path)
    
    # Use unified processor for complete processing
    processor = UnifiedVideoProcessor()
    success = processor.process_video_complete(
        video_path=opt.path,
        dataset_dir=base_dir,
        extract_audio_flag=True,
        asr_model="ave"
    )
    
    if not success:
        print("[ERROR] Video processing failed")
        exit(1)
    
    