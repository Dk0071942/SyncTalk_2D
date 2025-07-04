#!/usr/bin/env python3
"""
Data preprocessing script for SyncTalk_2D
Handles video processing, frame extraction, landmark detection, and audio feature extraction
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synctalk.utils.video_processor import UnifiedVideoProcessor
from synctalk.utils.preprocessing_utils import check_preprocessing_status, get_preprocessing_info, save_preprocessing_metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for SyncTalk_2D training')
    
    # Required arguments
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--name', type=str, required=True,
                        help='Name for the dataset (will create dataset/{name}/ directory)')
    
    # Audio model selection
    parser.add_argument('--asr_model', type=str, default='ave', choices=['ave', 'hubert', 'wenet'],
                        help='ASR model to use for audio feature extraction (default: ave)')
    
    # Optional arguments
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Base directory for datasets (default: dataset)')
    parser.add_argument('--fps', type=int, default=25,
                        help='Target FPS for video processing (default: 25)')
    parser.add_argument('--skip_frames', action='store_true',
                        help='Skip frame extraction if already done')
    parser.add_argument('--skip_landmarks', action='store_true',
                        help='Skip landmark detection if already done')
    parser.add_argument('--skip_audio', action='store_true',
                        help='Skip audio feature extraction if already done')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if data exists')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check preprocessing status without processing')
    
    args = parser.parse_args()
    
    # Validate input video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Create dataset directory
    dataset_path = os.path.join(args.dataset_dir, args.name)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Check existing preprocessing status
    status = check_preprocessing_status(dataset_path)
    
    print(f"=== Preprocessing video: {args.video_path} ===")
    print(f"Dataset name: {args.name}")
    print(f"ASR model: {args.asr_model}")
    print(f"Output directory: {dataset_path}")
    print()
    
    # Display current status using our utility
    print("Current preprocessing status:")
    print(get_preprocessing_info(dataset_path))
    print()
    
    # Check if already fully preprocessed
    if status['is_complete'] and not args.force:
        print("✅ Dataset is already fully preprocessed!")
        if args.check_only:
            return
        print("Use --force to reprocess anyway.")
        return
    
    if args.check_only:
        print("Preprocessing needed. Run without --check_only to process.")
        return
    
    # Auto-skip already completed steps unless forced
    if not args.force:
        details = status['details']
        if details['frames']['exists'] and not args.skip_frames:
            print("  → Frames already extracted, skipping (use --force to reprocess)")
            args.skip_frames = True
        if details['landmarks']['exists'] and not args.skip_landmarks:
            print("  → Landmarks already detected, skipping (use --force to reprocess)")
            args.skip_landmarks = True
        if details['audio']['exists'] and any(details['audio_features'].values()) and not args.skip_audio:
            print("  → Audio already processed, skipping (use --force to reprocess)")
            args.skip_audio = True
        print()
    
    # Run preprocessing
    try:
        # Use the unified video processor
        processor = UnifiedVideoProcessor()
        
        # Define progress callback for better user feedback
        def progress_callback(current, total, message):
            if current % 10 == 0 or current == total:
                print(f"  {message}")
        
        # Process the video
        print("Starting video preprocessing...")
        success = processor.process_video_complete(
            video_path=args.video_path,
            dataset_dir=dataset_path,
            extract_audio_flag=not args.skip_audio,
            asr_model=args.asr_model,
            progress_callback=progress_callback if not args.skip_frames else None,
            force=args.force,
            skip_frames=args.skip_frames,
            skip_landmarks=args.skip_landmarks,
            skip_audio=args.skip_audio
        )
        
        if not success:
            raise Exception("Video processing failed")
        
        print("\n=== Preprocessing completed successfully! ===")
        print(f"Preprocessed data saved to: {dataset_path}")
        
        # Save preprocessing metadata
        save_preprocessing_metadata(dataset_path, {
            'video_path': args.video_path,
            'name': args.name,
            'asr_model': args.asr_model,
            'fps': args.fps
        })
        
        # Display final status
        print("\nFinal preprocessing status:")
        print(get_preprocessing_info(dataset_path))
            
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()