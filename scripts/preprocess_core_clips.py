#!/usr/bin/env python3
"""
Preprocess core clips to extract frames and landmarks.
Creates a dataset-like structure for each core clip.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synctalk.processing.media_processor import MediaProcessor


class CoreClipsPreprocessor:
    """Preprocesses core clips to extract frames and landmarks."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize preprocessor.
        
        Args:
            model_name: Specific model to process, or None for all models
        """
        self.model_name = model_name
        self.core_clips_dir = Path("./core_clips")
        self.dataset_dir = Path("./dataset")
        self.video_processor = MediaProcessor(use_temp=False)
            
    def process_all_models(self):
        """Process core clips for all available models."""
        if not self.core_clips_dir.exists():
            print(f"Core clips directory not found: {self.core_clips_dir}")
            return
            
        # Get all model directories
        if self.model_name:
            model_dirs = [self.core_clips_dir / self.model_name]
        else:
            model_dirs = [d for d in self.core_clips_dir.iterdir() if d.is_dir()]
            
        for model_dir in model_dirs:
            if model_dir.exists():
                print(f"\nProcessing model: {model_dir.name}")
                self.process_model(model_dir.name)
            else:
                print(f"Model directory not found: {model_dir}")
                
    def process_model(self, model_name: str):
        """Process all clips for a specific model."""
        model_clips_dir = self.core_clips_dir / model_name
        model_dataset_dir = self.dataset_dir / model_name / "core_clips"
        
        if not model_clips_dir.exists():
            print(f"No clips found for model: {model_name}")
            return
            
        # Get all video files
        video_files = list(model_clips_dir.glob("*.mp4"))
        
        if not video_files:
            print(f"No video files found in {model_clips_dir}")
            return
            
        print(f"Found {len(video_files)} video files for {model_name}")
        
        # Process each video
        for video_file in tqdm(video_files, desc=f"Processing {model_name} clips", leave=False, position=0, ncols=100):
            self.process_video(video_file, model_dataset_dir)
            
    def process_video(self, video_path: Path, output_base_dir: Path):
        """
        Process a single video clip.
        
        Args:
            video_path: Path to the video file
            output_base_dir: Base directory for output
        """
        # Create clip-specific directory (use filename without extension)
        clip_name = video_path.stem
        clip_dir = output_base_dir / clip_name
        
        # Check if already processed
        full_body_dir = clip_dir / "full_body_img"
        landmarks_dir = clip_dir / "landmarks"
        
        if full_body_dir.exists() and landmarks_dir.exists():
            # Check if frames exist
            existing_frames = list(full_body_dir.glob("*.jpg"))
            existing_landmarks = list(landmarks_dir.glob("*.lms"))
            
            if existing_frames and existing_landmarks:
                print(f"\nSkipping {clip_name} - already processed")
                print(f"  Found {len(existing_frames)} frames and {len(existing_landmarks)} landmarks")
                return
                
        # Create clip directory
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {clip_name}")
        
        # Use unified processor - extract frames without audio
        frames_dir = str(full_body_dir)
        num_frames = self.video_processor.extract_frames(
            str(video_path), 
            frames_dir,
            convert_to_25fps=True
        )
        
        if num_frames == 0:
            print(f"  ERROR: No frames extracted from {clip_name}")
            return
            
        print(f"  Extracted {num_frames} frames")
        
        # Detect landmarks using unified processor
        landmarks_dir_str = str(landmarks_dir)
        success = self.video_processor.detect_landmarks(
            frames_dir,
            landmarks_dir_str
        )
        
        if not success:
            print(f"  ERROR: Failed to detect landmarks for {clip_name}")
            return
            
        # Count successful detections for info file
        landmarks_files = list(landmarks_dir.glob("*.lms"))
        successful_detections = len(landmarks_files)
        
        # Create info file
        info_path = clip_dir / "info.txt"
        with open(info_path, "w") as f:
            f.write(f"Source: {video_path.name}\n")
            f.write(f"FPS: 25\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Duration: {num_frames / 25.0:.2f}s\n")
            f.write(f"Landmarks detected: {successful_detections}/{num_frames}\n")
            
        print(f"  Saved info to {info_path}")
        
    def verify_processing(self):
        """Verify that all clips have been processed correctly."""
        if self.model_name:
            models = [self.model_name]
        else:
            models = [d.name for d in self.core_clips_dir.iterdir() if d.is_dir()]
            
        print("\nVerification Report:")
        print("=" * 60)
        
        for model in models:
            print(f"\nModel: {model}")
            model_clips_dir = self.core_clips_dir / model
            model_dataset_dir = self.dataset_dir / model / "core_clips"
            
            if not model_clips_dir.exists():
                print(f"  ❌ Source directory not found")
                continue
                
            video_files = list(model_clips_dir.glob("*.mp4"))
            print(f"  Source videos: {len(video_files)}")
            
            if model_dataset_dir.exists():
                processed_clips = [d for d in model_dataset_dir.iterdir() if d.is_dir()]
                print(f"  Processed clips: {len(processed_clips)}")
                
                for clip_dir in processed_clips:
                    frames = list((clip_dir / "full_body_img").glob("*.jpg"))
                    landmarks = list((clip_dir / "landmarks").glob("*.lms"))
                    
                    status = "✅" if frames and landmarks else "❌"
                    print(f"    {status} {clip_dir.name}: {len(frames)} frames, {len(landmarks)} landmarks")
            else:
                print(f"  ❌ No processed clips found")


def main():
    parser = argparse.ArgumentParser(description="Preprocess core clips for SyncTalk 2D")
    parser.add_argument("--model", type=str, help="Specific model to process (e.g., LS1)")
    parser.add_argument("--verify", action="store_true", help="Verify processing status")
    
    args = parser.parse_args()
    
    preprocessor = CoreClipsPreprocessor(model_name=args.model)
    
    if args.verify:
        preprocessor.verify_processing()
    else:
        preprocessor.process_all_models()
        print("\nProcessing complete!")
        preprocessor.verify_processing()


if __name__ == "__main__":
    main()