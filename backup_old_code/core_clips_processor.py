import os
import cv2
import torch
import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm

from core_clips_manager import CoreClipsManager
from frame_based_structures import CoreClip
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
                 clip_start_frame: int,
                 clip_end_frame: int,
                 padding_frames: int,
                 needs_lipsync: bool,
                 fps: int = 25):
        self.start_time = start_time  # In output timeline
        self.end_time = end_time
        self.clip = clip
        self.clip_start_frame = clip_start_frame  # In source clip (frame number)
        self.clip_end_frame = clip_end_frame  # In source clip (frame number)
        self.padding_frames = padding_frames  # Number of frames to pad with last frame
        self.needs_lipsync = needs_lipsync
        self.fps = fps
        
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
        
    @property
    def total_frames(self) -> int:
        """Total number of frames including padding."""
        return (self.clip_end_frame - self.clip_start_frame) + self.padding_frames
        
    def __repr__(self):
        return f"EDL({self.start_time:.2f}-{self.end_time:.2f}, {self.clip.clip_type}, frames={self.clip_start_frame}-{self.clip_end_frame}, pad={self.padding_frames}, lipsync={self.needs_lipsync})"


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
            
            segments = merged_segments
            
            # Visualize if requested
            if visualize_vad:
                vad_output = output_path.replace('.mp4', '_vad.png')
                self.vad.visualize_vad(audio_path, segments, vad_output)
                
            print(f"VAD detected {len(segments)} segments in {audio_duration:.2f}s audio (after merging same types)")
            
            # Print merged segments
            print("\nMerged segments:")
            for i, seg in enumerate(segments):
                print(f"  {i}: {seg.start_time:.3f}s - {seg.end_time:.3f}s ({seg.duration:.3f}s) - {seg.label}")
            
            # Step 2: Create Edit Decision List (EDL)
            if progress_callback:
                progress_callback(10, 100, "Creating edit decision list...")
                
            edl = self._create_edit_decision_list(segments)
            
            # Print detailed EDL information
            print("\n" + "="*80)
            print("EDIT DECISION LIST (EDL)")
            print("="*80)
            total_frames_by_clip = {}
            
            for idx, item in enumerate(edl):
                print(f"\nSegment {idx}:")
                print(f"  Time: {item.start_time:.3f}s - {item.end_time:.3f}s (duration: {item.duration:.3f}s)")
                print(f"  Type: {item.clip.clip_type} {'(with lip-sync)' if item.needs_lipsync else ''}")
                print(f"  Clip: {item.clip.path}")
                print(f"  Source frames: {item.clip_start_frame} - {item.clip_end_frame} ({item.clip_end_frame - item.clip_start_frame} frames)")
                print(f"  Padding frames: {item.padding_frames}")
                print(f"  Total output frames: {item.total_frames}")
                
                # Track total frames used from each clip
                clip_name = item.clip.path
                if clip_name not in total_frames_by_clip:
                    total_frames_by_clip[clip_name] = 0
                total_frames_by_clip[clip_name] += item.total_frames
            
            print("\n" + "-"*80)
            print("TOTAL FRAMES USED FROM EACH CORE VIDEO:")
            print("-"*80)
            for clip_name, frame_count in sorted(total_frames_by_clip.items()):
                print(f"  {clip_name}: {frame_count} frames ({frame_count/self.fps:.3f}s)")
            
            total_output_frames = sum(item.total_frames for item in edl)
            print(f"\nTotal output frames: {total_output_frames} ({total_output_frames/self.fps:.3f}s)")
            print("="*80 + "\n")
            
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
        
        # Keep track of the global timeline position
        global_time = 0.0
        
        for segment in segments:
            # Select clips for this segment using frame-based approach
            clip_selections = self.clips_manager.select_clips_for_segment(
                segment.duration, 
                "talk" if segment.label == "speech" else "silence",
                fps=self.fps
            )
            
            # Process each clip selection directly
            for clip, start_frame, end_frame, padding_frames in clip_selections:
                # Calculate duration based on total frames (including padding)
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
                
        return edl
        
    def _process_edl_item(self, edl_item: EditDecisionItem, 
                         audio_feats: np.ndarray, 
                         item_idx: int) -> str:
        """
        Process a single EDL item, applying lip-sync if needed.
        Plays full clip first, then pads with last frame if needed.
        
        Args:
            edl_item: The EDL item to process
            audio_feats: Audio features for the entire audio
            item_idx: Index of this item (for naming)
            
        Returns:
            Path to the processed segment video
        """
        print(f"\nProcessing EDL item {item_idx}:")
        print(f"  Clip: {edl_item.clip.path}")
        print(f"  Time range: {edl_item.start_time:.3f}s - {edl_item.end_time:.3f}s")
        
        # Extract frames and landmarks for this clip
        frames_dir, landmarks_dir, num_frames = self.clips_manager.extract_frames_and_landmarks(edl_item.clip)
        
        # Use frame-based information from EDL item
        start_frame = edl_item.clip_start_frame
        end_frame = edl_item.clip_end_frame
        clip_frames = end_frame - start_frame
        total_frames_needed = edl_item.total_frames
        
        print(f"  Source frames: {start_frame} - {end_frame} ({clip_frames} frames)")
        print(f"  Padding frames: {edl_item.padding_frames}")
        print(f"  Total output frames: {total_frames_needed}")
        
        # Create output video for this segment
        segment_path = os.path.join(self.temp_dir, f"segment_{item_idx:04d}.mp4")
        
        # Get first frame to determine dimensions
        first_frame_path = os.path.join(frames_dir, f"{start_frame}.jpg")
        if not os.path.exists(first_frame_path):
            print(f"Warning: Frame {start_frame} not found, trying frame 0")
            first_frame_path = os.path.join(frames_dir, "0.jpg")
            
        first_frame = cv2.imread(first_frame_path)
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(segment_path, fourcc, self.fps, (w, h))
        
        # Calculate the last frame index for padding
        last_frame_idx = end_frame - 1
        
        # Track frame usage
        frame_usage = []
        
        # Process frames
        for i in range(total_frames_needed):
            if i < clip_frames:
                # Play the clip normally
                frame_idx = start_frame + i
                frame_usage.append(f"{frame_idx}")
            else:
                # Pad with the last frame
                frame_idx = last_frame_idx
                frame_usage.append(f"{frame_idx}*")  # * indicates padding
            
            # Load frame and landmarks
            frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
            lms_path = os.path.join(landmarks_dir, f"{frame_idx}.lms")
            
            # Check if frame exists
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_idx} not found, using last valid frame")
                frame_idx = min(frame_idx, num_frames - 1)
                frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
                lms_path = os.path.join(landmarks_dir, f"{frame_idx}.lms")
            
            img = cv2.imread(frame_path)
            
            # Apply lip-sync if this is a speech segment
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
                
                # Calculate audio feature index based on output timeline
                audio_frame_idx = int(edl_item.start_time * self.fps) + i
                
                # Get audio features for this frame
                audio_feat = get_audio_features(audio_feats, audio_frame_idx).numpy()
                
                # Apply lip-sync with blending
                result_img = self._generate_frame_with_blending(img, audio_feat, lms)
            else:
                # For silence or first pass (no audio_feats), use frame as-is
                result_img = img
                
            out.write(result_img)
            
        out.release()
        
        # Print frame usage summary
        print(f"  Frame sequence: {frame_usage[0]} - {frame_usage[-1]}")
        if edl_item.padding_frames > 0:
            print(f"  Note: Last {edl_item.padding_frames} frames are padded (marked with *)")
        
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
        # For now, use the standard method without blending since we have 110 landmarks
        # The face blending utils were designed for 68 landmarks
        # TODO: Adapt face blending to work with 110 landmarks
        return self.sync_talk.generate_frame(img.copy(), audio_feat, lms, parsing=None)
        
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