"""
Core clips video processor for SyncTalk 2D.

This module implements the Core Clips mode that uses pre-recorded video segments
selected dynamically based on Voice Activity Detection (VAD) to generate output.
"""

import os
import cv2
import torch
import numpy as np
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Callable, Dict, Tuple
from tqdm import tqdm

# Import core components from synctalk package
from ..core.vad import SileroVAD, AudioSegment
from ..core.structures import EditDecisionItem
from ..core.clips_manager import CoreClipsManager
from ..utils.face_blending import blend_faces, create_face_mask

# Import SyncTalk inference temporarily (will be replaced by StandardVideoProcessor)
import sys
sys.path.append('.')
from inference_module import SyncTalkInference
from utils import get_audio_features


class CoreClipsProcessor:
    """
    Video processor using pre-recorded core clips selected by VAD.
    
    This processor analyzes audio with Voice Activity Detection to identify
    speech and silence segments, then assembles video using appropriate
    pre-recorded clips with lip-sync applied to speech segments.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the core clips processor.
        
        Args:
            model_name: Name of the model (e.g., "LS1")
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Initialize components
        self.clips_manager = CoreClipsManager(model_name)
        self.vad = SileroVAD()
        self.sync_talk: Optional[SyncTalkInference] = None
        
        # Configuration
        self.fps = 25
        self.temp_dir: Optional[str] = None
    
    def _analyze_audio_segments(self, audio_path: str, vad_threshold: float = 0.5,
                               min_silence_duration: float = 0.75) -> Tuple[List[AudioSegment], float]:
        """
        Analyze audio with VAD to detect speech and silence segments.
        
        Args:
            audio_path: Path to audio file
            vad_threshold: VAD threshold for speech detection
            min_silence_duration: Minimum silence duration to keep as silence
            
        Returns:
            Tuple of (segments list, audio duration)
        """
        # Configure VAD
        self.vad.threshold = vad_threshold
        
        # Process audio
        segments, audio_duration = self.vad.process_audio(audio_path)
        
        # Merge short segments to avoid choppy transitions
        segments = self.vad.merge_short_segments(segments, min_duration=min_silence_duration)
        
        # Merge consecutive segments of the same type
        merged_segments = []
        current_segment = None
        
        for segment in segments:
            if current_segment is None:
                current_segment = segment
            elif current_segment.label == segment.label:
                # Same type, merge them
                current_segment = AudioSegment(
                    start_time=current_segment.start_time,
                    end_time=segment.end_time,
                    label=current_segment.label
                )
            else:
                # Different type, save current and start new
                merged_segments.append(current_segment)
                current_segment = segment
        
        # Don't forget the last segment
        if current_segment:
            merged_segments.append(current_segment)
        
        print(f"\nVAD Analysis Results:")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Segments detected: {len(merged_segments)}")
        for i, seg in enumerate(merged_segments):
            print(f"    {i}: {seg.start_time:.3f}s - {seg.end_time:.3f}s ({seg.duration:.3f}s) - {seg.label}")
        
        return merged_segments, audio_duration
    
    def _create_edit_decision_list(self, segments: List[AudioSegment]) -> List[EditDecisionItem]:
        """
        Create an edit decision list mapping audio segments to video clips.
        
        Args:
            segments: List of audio segments from VAD
            
        Returns:
            List of edit decision items
        """
        edl = []
        global_time = 0.0
        
        for segment in segments:
            # Select clips for this segment using intelligent selection
            clip_selections = self.clips_manager.select_clips_for_segment(
                segment.duration, 
                "talk" if segment.label == "speech" else "silence",
                fps=self.fps
            )
            
            # Convert each selection to an EDL item
            for clip, start_frame, end_frame, padding_frames in clip_selections:
                total_frames = (end_frame - start_frame) + padding_frames
                duration = total_frames / self.fps
                
                edl_item = EditDecisionItem(
                    start_time=global_time,
                    end_time=global_time + duration,
                    clip=clip,
                    clip_start_frame=start_frame,
                    clip_end_frame=end_frame,
                    padding_frames=padding_frames,
                    needs_lipsync=(segment.label == "speech"),
                    fps=self.fps
                )
                
                edl.append(edl_item)
                global_time += duration
        
        # Validate EDL
        for item in edl:
            if not item.validate():
                print(f"Warning: Invalid EDL item: {item}")
        
        return edl
    
    def _process_speech_segment(self, edl_item: EditDecisionItem, audio_feats: np.ndarray,
                               item_idx: int) -> str:
        """
        Process a speech segment with lip-sync.
        
        Args:
            edl_item: The EDL item to process
            audio_feats: Audio features for the entire audio
            item_idx: Index for naming
            
        Returns:
            Path to processed segment video
        """
        # Extract frames and landmarks
        frames_dir, landmarks_dir, num_frames = self.clips_manager.extract_frames_and_landmarks(
            edl_item.clip
        )
        
        # Create output video
        segment_path = os.path.join(self.temp_dir, f"segment_{item_idx:04d}.mp4")
        
        # Get dimensions from first frame
        first_frame_path = os.path.join(frames_dir, f"{edl_item.clip_start_frame}.jpg")
        if not os.path.exists(first_frame_path):
            first_frame_path = os.path.join(frames_dir, "0.jpg")
        
        first_frame = cv2.imread(first_frame_path)
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(segment_path, fourcc, self.fps, (w, h))
        
        # Process frames
        clip_frames = edl_item.clip_end_frame - edl_item.clip_start_frame
        last_frame_idx = edl_item.clip_end_frame - 1
        
        for i in range(edl_item.total_frames):
            # Determine which frame to use
            if i < clip_frames:
                frame_idx = edl_item.clip_start_frame + i
            else:
                # Padding - use last frame
                frame_idx = last_frame_idx
            
            # Load frame and landmarks
            frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
            lms_path = os.path.join(landmarks_dir, f"{frame_idx}.lms")
            
            img = cv2.imread(frame_path)
            
            # Load landmarks
            lms_list = []
            with open(lms_path, "r") as f:
                for line in f:
                    coords = line.strip().split()
                    lms_list.append(np.array(coords, dtype=np.float32))
            lms = np.array(lms_list, dtype=np.int32)
            
            # Calculate audio feature index
            audio_frame_idx = int(edl_item.start_time * self.fps) + i
            audio_feat = get_audio_features(audio_feats, audio_frame_idx).numpy()
            
            # Generate lip-synced frame
            result_img = self.sync_talk.generate_frame(img.copy(), audio_feat, lms, parsing=None)
            
            writer.write(result_img)
        
        writer.release()
        return segment_path
    
    def _process_silence_segment(self, edl_item: EditDecisionItem, item_idx: int) -> str:
        """
        Process a silence segment without lip-sync.
        
        Args:
            edl_item: The EDL item to process
            item_idx: Index for naming
            
        Returns:
            Path to processed segment video
        """
        # Extract frames
        frames_dir, _, num_frames = self.clips_manager.extract_frames_and_landmarks(
            edl_item.clip
        )
        
        # Create output video
        segment_path = os.path.join(self.temp_dir, f"segment_{item_idx:04d}.mp4")
        
        # Get dimensions
        first_frame = cv2.imread(os.path.join(frames_dir, "0.jpg"))
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(segment_path, fourcc, self.fps, (w, h))
        
        # Process frames
        clip_frames = edl_item.clip_end_frame - edl_item.clip_start_frame
        last_frame_idx = edl_item.clip_end_frame - 1
        
        for i in range(edl_item.total_frames):
            # Determine which frame to use
            if i < clip_frames:
                frame_idx = edl_item.clip_start_frame + i
            else:
                # Padding - use last frame
                frame_idx = last_frame_idx
            
            # Load and write frame directly
            frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
            img = cv2.imread(frame_path)
            writer.write(img)
        
        writer.release()
        return segment_path
    
    def _assemble_final_video(self, segment_paths: List[str], audio_path: str,
                             output_path: str) -> None:
        """
        Assemble final video from segments and add audio.
        
        Args:
            segment_paths: List of segment video paths
            audio_path: Original audio path
            output_path: Final output path
        """
        # Create concat list
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")
        
        # Concatenate segments
        temp_video = os.path.join(self.temp_dir, "concatenated.mp4")
        concat_cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            temp_video
        ]
        subprocess.run(concat_cmd, check=True)
        
        # Add audio
        final_cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        subprocess.run(final_cmd, check=True)
    
    def generate_video(self, audio_path: str, output_path: str,
                      asr_mode: str = "ave",
                      vad_threshold: float = 0.5,
                      min_silence_duration: float = 0.75,
                      visualize_vad: bool = False,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Generate video using core clips with VAD-based selection.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output video
            asr_mode: Audio encoder mode ("ave", "hubert", "wenet")
            vad_threshold: VAD threshold for speech detection
            min_silence_duration: Minimum silence duration to keep as silence
            visualize_vad: Whether to save VAD visualization
            progress_callback: Optional progress callback(current, total, message)
            
        Returns:
            Path to the generated video file
        """
        try:
            # Create temp directory
            self.temp_dir = tempfile.mkdtemp(prefix="synctalk_core_")
            
            # Step 1: Analyze audio with VAD
            if progress_callback:
                progress_callback(0, 100, "Analyzing audio with VAD...")
            
            segments, audio_duration = self._analyze_audio_segments(
                audio_path, vad_threshold, min_silence_duration
            )
            
            # Visualize VAD if requested
            if visualize_vad:
                vad_output = output_path.replace('.mp4', '_vad.png')
                self.vad.visualize_vad(audio_path, segments, vad_output)
                print(f"VAD visualization saved to: {vad_output}")
            
            # Step 2: Create Edit Decision List
            if progress_callback:
                progress_callback(10, 100, "Creating edit decision list...")
            
            edl = self._create_edit_decision_list(segments)
            
            # Print EDL summary
            print(f"\nEdit Decision List Summary:")
            print(f"  Total segments: {len(edl)}")
            total_frames = sum(item.total_frames for item in edl)
            print(f"  Total frames: {total_frames} ({total_frames/self.fps:.2f}s)")
            
            # Step 3: Initialize lip-sync model
            if progress_callback:
                progress_callback(20, 100, "Loading lip-sync model...")
            
            self.sync_talk = SyncTalkInference(self.model_name, str(self.device))
            self.sync_talk.load_models(mode=asr_mode)
            
            # Process audio features
            audio_feats = self.sync_talk.process_audio(audio_path)
            
            # Step 4: Process each EDL item
            if progress_callback:
                progress_callback(30, 100, "Processing video segments...")
            
            processed_segments = []
            total_items = len(edl)
            
            for idx, edl_item in enumerate(edl):
                if progress_callback:
                    progress = 30 + (idx / total_items) * 50  # 30-80%
                    progress_callback(int(progress), 100, 
                                    f"Processing segment {idx+1}/{total_items}")
                
                # Process based on segment type
                if edl_item.needs_lipsync:
                    segment_path = self._process_speech_segment(
                        edl_item, audio_feats, idx
                    )
                else:
                    segment_path = self._process_silence_segment(edl_item, idx)
                
                processed_segments.append(segment_path)
            
            # Step 5: Assemble final video
            if progress_callback:
                progress_callback(80, 100, "Assembling final video...")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self._assemble_final_video(processed_segments, audio_path, output_path)
            
            if progress_callback:
                progress_callback(100, 100, "Done!")
            
            print(f"[INFO] Generated video saved to: {output_path}")
            return output_path
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def get_clip_statistics(self) -> Dict:
        """
        Get statistics about available clips.
        
        Returns:
            Dictionary with clip information
        """
        return self.clips_manager.get_clip_info()
    
    def run_cli(self, audio_path: str, output_dir: str = "./result",
                asr_mode: str = "ave", vad_threshold: float = 0.5,
                min_silence_duration: float = 0.75,
                visualize_vad: bool = False) -> str:
        """
        Run processor in CLI mode with progress bar.
        
        Args:
            audio_path: Path to input audio
            output_dir: Output directory
            asr_mode: Audio encoder mode
            vad_threshold: VAD threshold
            min_silence_duration: Minimum silence duration
            visualize_vad: Whether to visualize VAD
            
        Returns:
            Path to generated video
        """
        # Generate output filename
        audio_name = Path(audio_path).stem
        output_path = os.path.join(
            output_dir,
            f"{self.model_name}_{audio_name}_core_clips.mp4"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Create progress bar wrapper
        pbar = None
        
        def tqdm_callback(current, total, message):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Generating video")
            pbar.update(current - pbar.n)
            pbar.set_description(message)
            if current >= total and pbar is not None:
                pbar.close()
                pbar = None
        
        # Generate video
        return self.generate_video(
            audio_path=audio_path,
            output_path=output_path,
            asr_mode=asr_mode,
            vad_threshold=vad_threshold,
            min_silence_duration=min_silence_duration,
            visualize_vad=visualize_vad,
            progress_callback=tqdm_callback
        )