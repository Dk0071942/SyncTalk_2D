import os
import cv2
import torch
import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm

from core_clips_manager import CoreClipsManager, CoreClip
from vad_torch import SileroVAD, AudioSegment
from utils import get_audio_features
from inference_module import SyncTalkInference
from face_blending_utils import blend_faces, match_color_histogram, create_face_mask


class EditDecisionItem:
    """Represents a single edit decision for video assembly."""
    
    def __init__(self, 
                 start_time: float,
                 end_time: float,
                 clip: CoreClip,
                 clip_start: float,
                 clip_end: float,
                 needs_lipsync: bool):
        self.start_time = start_time  # In output timeline
        self.end_time = end_time
        self.clip = clip
        self.clip_start = clip_start  # In source clip
        self.clip_end = clip_end
        self.needs_lipsync = needs_lipsync
        
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
        
    def __repr__(self):
        return f"EDL({self.start_time:.2f}-{self.end_time:.2f}, {self.clip.clip_type}, lipsync={self.needs_lipsync})"


class CoreClipsProcessor:
    """Processes video generation using core clips with VAD-based selection."""
    
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize the processor.
        
        Args:
            model_name: Name of the model (e.g., "LS1")
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.clips_manager = CoreClipsManager(model_name)
        self.vad = SileroVAD()
        self.sync_talk = None  # Will be initialized when needed
        
        # Output settings
        self.fps = 25
        self.temp_dir = None
        
    def generate_video(self,
                      audio_path: str,
                      output_path: str,
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
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated video file
        """
        try:
            # Create temp directory
            self.temp_dir = tempfile.mkdtemp(prefix="synctalk_core_")
            
            # Step 1: Apply VAD to audio
            if progress_callback:
                progress_callback(0, 100, "Analyzing audio with VAD...")
                
            self.vad.threshold = vad_threshold
            segments, audio_duration = self.vad.process_audio(audio_path)
            
            # Merge short segments
            segments = self.vad.merge_short_segments(segments, min_duration=min_silence_duration)
            
            # Visualize if requested
            if visualize_vad:
                vad_output = output_path.replace('.mp4', '_vad.png')
                self.vad.visualize_vad(audio_path, segments, vad_output)
                
            print(f"VAD detected {len(segments)} segments in {audio_duration:.2f}s audio")
            
            # Step 2: Create Edit Decision List (EDL)
            if progress_callback:
                progress_callback(10, 100, "Creating edit decision list...")
                
            edl = self._create_edit_decision_list(segments)
            
            # Step 3: Initialize SyncTalk model for lip-sync
            if progress_callback:
                progress_callback(20, 100, "Loading lip-sync model...")
                
            self.sync_talk = SyncTalkInference(self.model_name, self.device)
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
                    progress_callback(int(progress), 100, f"Processing segment {idx+1}/{total_items}")
                    
                segment_path = self._process_edl_item(edl_item, audio_feats, idx)
                processed_segments.append(segment_path)
                
            # Step 5: Concatenate all segments
            if progress_callback:
                progress_callback(80, 100, "Assembling final video...")
                
            temp_video = os.path.join(self.temp_dir, "concatenated.mp4")
            self._concatenate_segments(processed_segments, temp_video)
            
            # Step 6: Add audio
            if progress_callback:
                progress_callback(90, 100, "Adding audio track...")
                
            self._add_audio(temp_video, audio_path, output_path)
            
            if progress_callback:
                progress_callback(100, 100, "Done!")
                
            print(f"[INFO] Generated video saved to: {output_path}")
            
            return output_path
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
    def _create_edit_decision_list(self, segments: List[AudioSegment]) -> List[EditDecisionItem]:
        """
        Create an edit decision list mapping audio segments to video clips.
        
        Args:
            segments: List of audio segments from VAD
            
        Returns:
            List of edit decision items
        """
        edl = []
        
        for segment in segments:
            # Select clips for this segment
            clip_selections = self.clips_manager.select_clips_for_segment(
                segment.duration, 
                "talk" if segment.label == "speech" else "silence"
            )
            
            # Create EDL items
            current_time = segment.start_time
            
            for clip, clip_start, clip_end in clip_selections:
                duration = clip_end - clip_start
                
                edl_item = EditDecisionItem(
                    start_time=current_time,
                    end_time=current_time + duration,
                    clip=clip,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    needs_lipsync=(segment.label == "speech")
                )
                
                edl.append(edl_item)
                current_time += duration
                
        return edl
        
    def _process_edl_item(self, edl_item: EditDecisionItem, 
                         audio_feats: np.ndarray, 
                         item_idx: int) -> str:
        """
        Process a single EDL item, applying lip-sync if needed.
        
        Args:
            edl_item: The EDL item to process
            audio_feats: Audio features for the entire audio
            item_idx: Index of this item (for naming)
            
        Returns:
            Path to the processed segment video
        """
        # Extract frames and landmarks for this clip
        frames_dir, landmarks_dir, num_frames = self.clips_manager.extract_frames_and_landmarks(edl_item.clip)
        
        # Calculate frame range for this segment
        start_frame = int(edl_item.clip_start * self.fps)
        end_frame = int(edl_item.clip_end * self.fps)
        
        # Handle looping if segment is longer than clip
        total_frames_needed = int(edl_item.duration * self.fps)
        clip_frames = end_frame - start_frame
        
        # Create output video for this segment
        segment_path = os.path.join(self.temp_dir, f"segment_{item_idx:04d}.mp4")
        
        # Get first frame to determine dimensions
        first_frame = cv2.imread(os.path.join(frames_dir, f"{start_frame}.jpg"))
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(segment_path, fourcc, self.fps, (w, h))
        
        # Process frames
        for i in range(total_frames_needed):
            # Calculate which frame to use (with looping)
            frame_idx = start_frame + (i % clip_frames)
            
            # Load frame and landmarks
            frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
            lms_path = os.path.join(landmarks_dir, f"{frame_idx}.lms")
            
            img = cv2.imread(frame_path)
            
            if edl_item.needs_lipsync:
                # Load landmarks
                lms_list = []
                with open(lms_path, "r") as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        arr = line.split(" ")
                        arr = np.array(arr, dtype=np.float32)
                        lms_list.append(arr)
                lms = np.array(lms_list, dtype=np.int32)
                
                # Calculate audio feature index
                audio_frame_idx = int(edl_item.start_time * self.fps) + i
                
                # Get audio features for this frame
                audio_feat = get_audio_features(audio_feats, audio_frame_idx).numpy()
                
                # Apply lip-sync with blending
                result_img = self._generate_frame_with_blending(img, audio_feat, lms)
            else:
                # For silence, use frame as-is
                result_img = img
                
            out.write(result_img)
            
        out.release()
        
        return segment_path
        
    def _generate_frame_with_blending(self, img: np.ndarray, audio_feat: np.ndarray, 
                                     lms: np.ndarray) -> np.ndarray:
        """
        Generate a frame with proper face blending to avoid neck/head boundary issues.
        
        Args:
            img: Original image
            audio_feat: Audio features
            lms: Facial landmarks
            
        Returns:
            Image with blended lip-synced face
        """
        try:
            # First, generate the full frame using standard method
            generated_full = self.sync_talk.generate_frame(img.copy(), audio_feat, lms, parsing=None)
            
            # Validate landmarks
            if len(lms) != 68:
                print(f"Warning: Expected 68 landmarks, got {len(lms)}. Using standard method.")
                return generated_full
                
            # Create a smooth face mask
            face_mask = create_face_mask(lms, img.shape, expansion_ratio=1.15)
            
            # Match colors between generated face and original
            # This helps reduce color discontinuity
            matched_generated = match_color_histogram(generated_full, img, face_mask)
            
            # Blend the generated face with original using smooth mask
            result = blend_faces(img, matched_generated, lms, feather_amount=30)
            
            return result
            
        except Exception as e:
            print(f"Warning: Face blending failed: {e}. Using standard method.")
            # Fallback to standard method without blending
            return self.sync_talk.generate_frame(img, audio_feat, lms, parsing=None)
        
    def _concatenate_segments(self, segment_paths: List[str], output_path: str):
        """
        Concatenate video segments using ffmpeg.
        
        Args:
            segment_paths: List of paths to segment videos
            output_path: Output video path
        """
        # Create concat file
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")
                
        # Use ffmpeg to concatenate
        cmd = f'ffmpeg -y -v error -f concat -safe 0 -i "{concat_file}" -c copy "{output_path}"'
        os.system(cmd)
        
    def _add_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Add audio track to video.
        
        Args:
            video_path: Path to video without audio
            audio_path: Path to audio file
            output_path: Final output path
        """
        cmd = f'ffmpeg -y -v error -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{output_path}"'
        os.system(cmd)
        
    def get_clip_statistics(self) -> Dict:
        """Get statistics about available clips."""
        return self.clips_manager.get_clip_info()


def test_core_clips_processor():
    """Test the core clips processor."""
    # Check if test audio exists
    test_audio = "pause_audio_test_23s.wav"
    
    if os.path.exists(test_audio):
        processor = CoreClipsProcessor("LS1")
        
        # Get clip info
        info = processor.get_clip_statistics()
        print("Available clips:")
        print(f"  Talk clips: {len(info['talk_clips'])}")
        print(f"  Silence clips: {len(info['silence_clips'])}")
        
        # Generate test video
        output_path = "test_core_clips_output.mp4"
        processor.generate_video(
            audio_path=test_audio,
            output_path=output_path,
            visualize_vad=True
        )
        
        print(f"Test video generated: {output_path}")
    else:
        print(f"Test audio not found: {test_audio}")


if __name__ == "__main__":
    test_core_clips_processor()