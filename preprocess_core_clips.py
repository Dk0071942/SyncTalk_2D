#!/usr/bin/env python3
"""
Preprocess core clips to extract frames and landmarks.
Creates a dataset-like structure for each core clip.
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import sys
sys.path.append('./data_utils')

from data_utils.get_landmark import Landmark


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
        self.landmark_detector = None
        
    def _init_landmark_detector(self):
        """Initialize landmark detector lazily."""
        if self.landmark_detector is None:
            print("Initializing landmark detector...")
            self.landmark_detector = Landmark()
            
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
        for video_file in tqdm(video_files, desc=f"Processing {model_name} clips"):
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
                
        # Create directories
        full_body_dir.mkdir(parents=True, exist_ok=True)
        landmarks_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {clip_name}")
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video info: {fps}fps, {total_frames} frames")
        
        # Check if we need to convert to 25fps
        temp_video = None
        if abs(fps - 25.0) > 0.1:
            print(f"  Converting from {fps}fps to 25fps...")
            temp_video = clip_dir / "temp_25fps.mp4"
            cmd = f'ffmpeg -y -v error -i "{video_path}" -vf "fps=25" -c:v libx264 "{temp_video}"'
            os.system(cmd)
            cap.release()
            cap = cv2.VideoCapture(str(temp_video))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        # Extract frames
        frame_count = 0
        frames_extracted = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = full_body_dir / f"{frame_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames_extracted.append(frame_path)
            frame_count += 1
            
        cap.release()
        
        # Remove temp video if created
        if temp_video and temp_video.exists():
            os.remove(temp_video)
            
        print(f"  Extracted {frame_count} frames")
        
        # Detect landmarks
        self._init_landmark_detector()
        
        successful_detections = 0
        last_valid_landmarks = None
        
        for i in tqdm(range(frame_count), desc="  Detecting landmarks", leave=False):
            frame_path = full_body_dir / f"{i}.jpg"
            lms_path = landmarks_dir / f"{i}.lms"
            
            try:
                # Detect landmarks
                pre_landmark, x1, y1 = self.landmark_detector.detect(str(frame_path))
                
                if pre_landmark is not None:
                    # Save landmarks
                    lms_lines = []
                    for p in pre_landmark:
                        x, y = p[0] + x1, p[1] + y1
                        lms_lines.append(f"{x} {y}")
                    
                    landmarks_content = "\n".join(lms_lines) + "\n"
                    with open(lms_path, "w") as f:
                        f.write(landmarks_content)
                    
                    last_valid_landmarks = landmarks_content
                    successful_detections += 1
                else:
                    # Use last valid landmarks if available
                    if last_valid_landmarks:
                        with open(lms_path, "w") as f:
                            f.write(last_valid_landmarks)
                    else:
                        print(f"\n  Warning: No landmarks detected for frame {i}")
                        
            except Exception as e:
                print(f"\n  Error detecting landmarks for frame {i}: {e}")
                # Use last valid landmarks if available
                if last_valid_landmarks:
                    with open(lms_path, "w") as f:
                        f.write(last_valid_landmarks)
                        
        print(f"  Successfully detected landmarks in {successful_detections}/{frame_count} frames")
        
        # Create info file
        info_path = clip_dir / "info.txt"
        with open(info_path, "w") as f:
            f.write(f"Source: {video_path.name}\n")
            f.write(f"FPS: 25\n")
            f.write(f"Frames: {frame_count}\n")
            f.write(f"Duration: {frame_count / 25.0:.2f}s\n")
            f.write(f"Original FPS: {fps}\n")
            f.write(f"Landmarks detected: {successful_detections}/{frame_count}\n")
            
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