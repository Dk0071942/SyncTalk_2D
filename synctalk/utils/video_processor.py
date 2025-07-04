"""
Unified video processing module for SyncTalk 2D.

This module provides a single implementation for video processing tasks:
- Frame extraction
- Frame rate conversion to 25fps
- Facial landmark detection
- Audio extraction

Used by all preprocessing scripts to avoid code duplication.
"""

import os
import cv2
import subprocess
from typing import Optional, Tuple, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from .ffmpeg_utils import FFmpegConfig, convert_fps as ffmpeg_convert_fps, extract_audio as ffmpeg_extract_audio


class UnifiedVideoProcessor:
    """Unified video processor for all preprocessing needs."""
    
    def __init__(self):
        """Initialize video processor."""
        self.landmark_detector = None
    
    def _save_state(self, dataset_dir: str, key: str, value: bool, metadata: dict = None):
        """Save preprocessing state."""
        state_file = os.path.join(dataset_dir, '.training_state.json')
        state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
            except:
                pass
        
        if 'preprocessing' not in state:
            state['preprocessing'] = {'completed': False, 'steps': {}}
        
        state['preprocessing']['steps'][key] = value
        if metadata:
            state['preprocessing'][key + '_metadata'] = metadata
        
        # Check if all steps completed
        steps = state['preprocessing']['steps']
        if all(steps.get(k, False) for k in ['frames', 'landmarks', 'audio', 'audio_features']):
            state['preprocessing']['completed'] = True
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
    def _init_landmark_detector(self):
        """Initialize landmark detector lazily."""
        if self.landmark_detector is None:
            import sys
            sys.path.append('./data_utils')
            from data_utils.get_landmark import Landmark
            self.landmark_detector = Landmark()
    
    def extract_audio(self, video_path: str, output_path: str, sample_rate: int = 16000, force: bool = False) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video
            output_path: Path for output WAV file
            sample_rate: Audio sample rate (default: 16000)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if audio already exists
        if os.path.exists(output_path) and not force:
            print(f'[INFO] Audio already extracted at {output_path}. Skipping.')
            return True
        
        print(f'[INFO] Extracting audio from {video_path} to {output_path}')
        
        try:
            result = ffmpeg_extract_audio(video_path, output_path, sample_rate)
            if result.returncode != 0:
                print(f'[ERROR] Failed to extract audio: {result.stderr}')
                return False
            print(f'[INFO] Audio extracted successfully')
            return True
        except Exception as e:
            print(f'[ERROR] Failed to extract audio: {e}')
            return False
    
    def extract_frames(self, 
                      video_path: str, 
                      output_dir: str,
                      convert_to_25fps: bool = True,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None,
                      force: bool = False) -> int:
        """
        Extract frames from video, converting to 25fps if needed.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            convert_to_25fps: Whether to convert to 25fps
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Number of frames extracted
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {video_path}")
            return 0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if frames already extracted
        existing_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        if existing_frames >= total_frames and total_frames > 0 and not force:
            print(f'[INFO] Frames already extracted ({existing_frames} frames). Skipping.')
            cap.release()
            return existing_frames
        elif existing_frames > 0 and force:
            print(f'[INFO] Found {existing_frames} existing frames, but force=True, re-extracting...')
        
        # Convert to 25fps if needed
        video_to_process = video_path
        temp_video = None
        
        if convert_to_25fps and abs(fps - 25.0) > 0.1:
            cap.release()
            print(f"[INFO] Converting video from {fps}fps to 25fps...")
            
            # Create temp video in the same directory as output
            temp_video = os.path.join(os.path.dirname(video_path), 
                                     os.path.basename(video_path).replace('.mp4', '_25fps.mp4'))
            
            try:
                result = ffmpeg_convert_fps(video_path, temp_video, target_fps=25)
                if result.returncode != 0:
                    print(f"[ERROR] Failed to convert video to 25fps: {result.stderr}")
                    cap = cv2.VideoCapture(video_path)  # Fall back to original
                else:
                    video_to_process = temp_video
                    cap = cv2.VideoCapture(temp_video)
                    if not cap.isOpened():
                        print("[ERROR] Failed to open converted video")
                        return 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            except Exception as e:
                print(f"[ERROR] Failed to convert video to 25fps: {e}")
                cap = cv2.VideoCapture(video_path)  # Fall back to original
        
        # Extract frames
        print(f"[INFO] Extracting {total_frames} frames...")
        frame_count = 0
        
        pbar = tqdm(total=total_frames, desc="Extracting frames", leave=False, position=0, ncols=100)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(output_dir, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
            pbar.update(1)
            
            if progress_callback:
                progress_callback(frame_count, total_frames, 
                                f"Extracting frame {frame_count}/{total_frames}")
        
        pbar.close()
        cap.release()
        
        # Clean up temp video if created
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
        
        print(f"[INFO] Extracted {frame_count} frames")
        return frame_count
    
    def detect_landmarks(self,
                        frames_dir: str,
                        output_dir: str,
                        progress_callback: Optional[Callable[[int, int, str], None]] = None,
                        force: bool = False) -> bool:
        """
        Detect landmarks for all frames in a directory.
        
        Args:
            frames_dir: Directory containing frame images
            output_dir: Directory to save landmark files
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            True if successful, False otherwise
        """
        os.makedirs(output_dir, exist_ok=True)
        self._init_landmark_detector()
        
        # Get frame files
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )
        
        if not frame_files:
            print("[ERROR] No frames found to process")
            return False
        
        print(f"[INFO] Detecting landmarks for {len(frame_files)} frames...")
        
        last_landmarks = None
        successful_detections = 0
        
        pbar = tqdm(total=len(frame_files), desc="Detecting landmarks", leave=False, position=0, ncols=100)
        
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            lms_path = os.path.join(output_dir, frame_file.replace('.jpg', '.lms'))
            
            # Skip if already exists
            if os.path.exists(lms_path) and not force:
                with open(lms_path, 'r') as f:
                    last_landmarks = f.read()
                successful_detections += 1
                pbar.update(1)
                continue
            
            try:
                pre_landmark, x1, y1 = self.landmark_detector.detect(frame_path)
                
                if pre_landmark is not None:
                    # Save landmarks
                    lms_lines = []
                    for p in pre_landmark:
                        x, y = p[0] + x1, p[1] + y1
                        lms_lines.append(f"{x} {y}")
                    
                    landmarks_content = "\n".join(lms_lines) + "\n"
                    with open(lms_path, "w") as f:
                        f.write(landmarks_content)
                    
                    last_landmarks = landmarks_content
                    successful_detections += 1
                else:
                    # Use last valid landmarks if available
                    if last_landmarks:
                        with open(lms_path, "w") as f:
                            f.write(last_landmarks)
                    else:
                        print(f"[WARNING] No landmarks detected for frame {idx}")
                        
            except Exception as e:
                print(f"[ERROR] Failed to detect landmarks for frame {idx}: {e}")
                # Use last valid landmarks if available
                if last_landmarks:
                    with open(lms_path, "w") as f:
                        f.write(last_landmarks)
            
            pbar.update(1)
            
            if progress_callback:
                progress_callback(idx + 1, len(frame_files), 
                                f"Detecting landmarks {idx + 1}/{len(frame_files)}")
        
        pbar.close()
        print(f"[INFO] Successfully detected landmarks for {successful_detections}/{len(frame_files)} frames")
        
        return successful_detections > 0
    
    def process_video_complete(self,
                             video_path: str,
                             dataset_dir: str,
                             extract_audio_flag: bool = True,
                             asr_model: str = "ave",
                             progress_callback: Optional[Callable[[int, int, str], None]] = None,
                             force: bool = False,
                             skip_frames: bool = False,
                             skip_landmarks: bool = False,
                             skip_audio: bool = False) -> bool:
        """
        Complete video processing pipeline.
        
        Creates the standard directory structure:
        - dataset_dir/full_body_img/ - Extracted frames
        - dataset_dir/landmarks/ - Facial landmarks
        - dataset_dir/aud.wav - Extracted audio
        
        Args:
            video_path: Path to input video
            dataset_dir: Base directory for dataset
            extract_audio_flag: Whether to extract audio
            asr_model: ASR model to use (for future compatibility)
            progress_callback: Optional progress callback
            
        Returns:
            True if successful, False otherwise
        """
        # Create directories
        frames_dir = os.path.join(dataset_dir, "full_body_img")
        landmarks_dir = os.path.join(dataset_dir, "landmarks")
        
        # Extract audio
        if extract_audio_flag and not skip_audio:
            audio_path = os.path.join(dataset_dir, "aud.wav")
            if not self.extract_audio(video_path, audio_path, force=force):
                return False
            self._save_state(dataset_dir, 'audio', True)
        
        # Extract frames
        if not skip_frames:
            num_frames = self.extract_frames(video_path, frames_dir, convert_to_25fps=True, 
                                           progress_callback=progress_callback, force=force)
            if num_frames == 0:
                return False
            self._save_state(dataset_dir, 'frames', True, {'count': num_frames})
        else:
            # Just count existing frames
            num_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            print(f"[INFO] Skipping frame extraction, found {num_frames} existing frames")
            if num_frames > 0:
                self._save_state(dataset_dir, 'frames', True, {'count': num_frames})
        
        # Detect landmarks
        if not skip_landmarks:
            if not self.detect_landmarks(frames_dir, landmarks_dir, progress_callback, force=force):
                return False
            landmark_count = len([f for f in os.listdir(landmarks_dir) if f.endswith('.lms')])
            self._save_state(dataset_dir, 'landmarks', True, {'count': landmark_count})
        else:
            print("[INFO] Skipping landmark detection")
            landmark_count = len([f for f in os.listdir(landmarks_dir) if f.endswith('.lms')])
            if landmark_count > 0:
                self._save_state(dataset_dir, 'landmarks', True, {'count': landmark_count})
        
        # Extract audio features (call existing script for now)
        if extract_audio_flag and not skip_audio:
            audio_path = os.path.join(dataset_dir, "aud.wav")
            audio_feature_files = {
                'ave': 'aud_ave.npy',
                'hubert': 'aud_hubert.npy',
                'wenet': 'aud_wenet.npy'
            }
            feature_file = audio_feature_files.get(asr_model, 'aud_ave.npy')
            feature_path = os.path.join(dataset_dir, feature_file)
            
            if os.path.exists(feature_path) and not force:
                print(f"[INFO] Audio features already extracted at {feature_path}. Skipping.")
            elif os.path.exists(audio_path):
                print(f"[INFO] Extracting {asr_model} audio features...")
                if asr_model == "ave":
                    cmd = f"python ./data_utils/ave/test_w2l_audio.py --wav_path {audio_path}"
                elif asr_model == "hubert":
                    cmd = f"python ./data_utils/hubert/test_hubert_audio.py --wav_path {audio_path}"
                elif asr_model == "wenet":
                    cmd = f"python ./data_utils/wenet/test_wenet_audio.py --wav_path {audio_path}"
                else:
                    print(f"[WARNING] Unknown ASR model: {asr_model}")
                    return True
                    
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                    print(f"[INFO] Audio features extracted to {feature_path}")
                    self._save_state(dataset_dir, 'audio_features', True, {'asr_model': asr_model})
                except subprocess.CalledProcessError as e:
                    print(f"[WARNING] Failed to extract audio features: {e}")
        
        return True


# Convenience function for backward compatibility
def process_video(video_path: str, dataset_dir: str, asr_model: str = "ave") -> bool:
    """
    Process a video file completely (backward compatibility wrapper).
    
    Args:
        video_path: Path to video file
        dataset_dir: Dataset directory to save outputs
        asr_model: ASR model to use
        
    Returns:
        True if successful
    """
    processor = UnifiedVideoProcessor()
    return processor.process_video_complete(video_path, dataset_dir, 
                                          extract_audio_flag=True, asr_model=asr_model)