"""
Standard video processor for SyncTalk 2D.

This module implements the standard video generation mode that uses
pre-extracted frames from training videos to generate lip-synced output.
"""

import os
import cv2
import torch
import numpy as np
import subprocess
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, Callable, Tuple, Dict, Any
from tqdm import tqdm

# Import required models and utilities
import sys
sys.path.append('.')
from synctalk.core.unet_328 import Model
from synctalk.core.utils import AudioEncoder, AudDataset, get_audio_features


class StandardVideoProcessor:
    """
    Standard video processor for lip-sync generation using pre-extracted frames.
    
    This processor uses a U-Net model to generate lip-synced faces by combining
    reference frames with audio features.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the standard video processor.
        
        Args:
            model_name: Name of the model (e.g., "AD2.2", "LS1")
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Set up paths
        self.checkpoint_path = Path(f"./checkpoint/{model_name}")
        self.dataset_dir = Path(f"./dataset/{model_name}")
        self.img_dir = self.dataset_dir / "full_body_img"
        self.lms_dir = self.dataset_dir / "landmarks"
        self.parsing_dir = self.dataset_dir / "parsing"
        
        # Models (loaded lazily)
        self.audio_encoder: Optional[AudioEncoder] = None
        self.unet_model: Optional[Model] = None
        
        # Configuration
        self.mode: Optional[str] = None
        self.fps: int = 25  # Default FPS, adjusted based on mode
        self.checkpoint_file: Optional[str] = None
    
    def load_models(self, checkpoint_number: Optional[int] = None, 
                    mode: str = "ave") -> str:
        """
        Load the audio encoder and U-Net models.
        
        Args:
            checkpoint_number: Specific checkpoint to load. If None, loads latest.
            mode: Audio feature mode ("ave", "hubert", or "wenet")
            
        Returns:
            Path to loaded checkpoint file
        """
        # Load audio encoder
        self.audio_encoder = AudioEncoder().to(self.device).eval()
        audio_ckpt_path = 'model/checkpoints/audio_visual_encoder.pth'
        ckpt = torch.load(audio_ckpt_path, map_location=self.device)
        self.audio_encoder.load_state_dict({
            f'audio_encoder.{k}': v for k, v in ckpt.items()
        })
        
        # Find checkpoint file
        if checkpoint_number is None:
            # Get the latest checkpoint
            checkpoints = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pth')],
                key=lambda x: int(x.split(".")[0])
            )
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {self.checkpoint_path}")
            checkpoint_file = self.checkpoint_path / checkpoints[-1]
        else:
            checkpoint_file = self.checkpoint_path / f"{checkpoint_number}.pth"
        
        # Load U-Net model
        self.unet_model = Model(6, mode).to(self.device)
        self.unet_model.load_state_dict(
            torch.load(checkpoint_file, map_location=self.device)
        )
        self.unet_model.eval()
        
        # Store configuration
        self.mode = mode
        self.checkpoint_file = str(checkpoint_file)
        
        # Adjust FPS based on mode
        self.fps = 20 if mode == "wenet" else 25
        
        print(f"Loaded models with checkpoint: {checkpoint_file}")
        return str(checkpoint_file)
    
    def _prepare_audio_features(self, audio_path: str) -> np.ndarray:
        """
        Process audio file and extract features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Audio features as numpy array
        """
        dataset = AudDataset(audio_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        outputs = []
        for mel in data_loader:
            mel = mel.to(self.device)
            with torch.no_grad():
                out = self.audio_encoder(mel)
            outputs.append(out)
        
        # Concatenate and pad with first/last frames
        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        
        # Pad with first/last frames for boundary handling
        audio_feats = torch.cat([
            first_frame.repeat(1, 1),
            outputs,
            last_frame.repeat(1, 1)
        ], dim=0).numpy()
        
        return audio_feats
    
    def _load_frame_data(self, frame_idx: int, img_dir: Path, lms_dir: Path, 
                        parsing_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load frame data including image, landmarks, and optional parsing.
        
        Args:
            frame_idx: Frame index
            img_dir: Directory containing images
            lms_dir: Directory containing landmarks
            parsing_dir: Optional directory containing parsing masks
            
        Returns:
            Tuple of (image, landmarks, parsing_mask)
        """
        # Load image
        img_path = img_dir / f"{frame_idx}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Load landmarks
        lms_path = lms_dir / f"{frame_idx}.lms"
        lms_list = []
        with open(lms_path, "r") as f:
            for line in f:
                coords = line.strip().split()
                lms_list.append(np.array(coords, dtype=np.float32))
        lms = np.array(lms_list, dtype=np.int32)
        
        # Load parsing if available
        parsing = None
        if parsing_dir and parsing_dir.exists():
            parsing_path = parsing_dir / f"{frame_idx}.png"
            if parsing_path.exists():
                parsing = cv2.imread(str(parsing_path))
        
        return img, lms, parsing
    
    def _generate_single_frame(self, img: np.ndarray, audio_feat: np.ndarray,
                              lms: np.ndarray, parsing: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a single frame using the U-Net model.
        
        Args:
            img: Input image
            audio_feat: Audio features for this frame
            lms: Facial landmarks (68 points)
            parsing: Optional parsing mask
            
        Returns:
            Generated frame with lip-sync
        """
        # Extract face region using landmarks
        xmin = lms[1][0]  # Left face boundary
        ymin = lms[52][1]  # Upper lip
        xmax = lms[31][0]  # Right face boundary
        width = xmax - xmin
        ymax = ymin + width  # Square crop
        
        # Crop face region
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img_backup = crop_img.copy()
        h, w = crop_img.shape[:2]
        
        # Crop parsing if available
        crop_parsing = None
        if parsing is not None:
            crop_parsing = parsing[ymin:ymax, xmin:xmax]
        
        # Resize to model input size
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()
        
        # Extract center region (320x320)
        img_real = crop_img[4:324, 4:324].copy()
        img_real_ori = img_real.copy()
        
        # Create masked input (black rectangle over mouth area)
        img_masked = cv2.rectangle(img_real_ori.copy(), (5, 5, 310, 305), (0, 0, 0), -1)
        
        # Convert to tensors (CHW format, normalized)
        img_real = img_real.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        img_real_T = torch.from_numpy(img_real)
        img_masked_T = torch.from_numpy(img_masked)
        img_concat_T = torch.cat([img_real_T, img_masked_T], dim=0).unsqueeze(0)
        
        # Reshape audio features based on mode
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)
        elif self.mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        elif self.mode == "ave":
            audio_feat = audio_feat.reshape(32, 16, 16)
        
        audio_feat_T = torch.from_numpy(audio_feat).unsqueeze(0).to(self.device)
        img_concat_T = img_concat_T.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            pred = self.unet_model(img_concat_T, audio_feat_T)[0]
        
        # Convert back to image
        pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)
        
        # Put prediction back into crop
        crop_img_ori[4:324, 4:324] = pred
        
        # Resize back to original crop size
        crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply parsing mask if available
        if crop_parsing is not None:
            # Preserve non-face regions (background, hair, etc.)
            parsing_mask = (
                (crop_parsing == [0, 0, 255]).all(axis=2) |  # Hair
                (crop_parsing == [255, 255, 255]).all(axis=2)  # Background
            )
            crop_img_ori[parsing_mask] = crop_img_backup[parsing_mask]
        
        # Put back into original image
        result_img = img.copy()
        result_img[ymin:ymax, xmin:xmax] = crop_img_ori
        
        return result_img
    
    def _write_video_file(self, frames: list, output_path: str, 
                         audio_path: str) -> None:
        """
        Write frames to video file and merge with audio.
        
        Args:
            frames: List of generated frames
            output_path: Output video path
            audio_path: Audio file to merge
        """
        if not frames:
            raise ValueError("No frames to write")
        
        # Get dimensions from first frame
        h, w = frames[0].shape[:2]
        
        # Write temporary video
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_output, fourcc, self.fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        writer.release()
        
        # Merge with audio using ffmpeg
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-nostats',
            '-i', temp_output,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', '20',
            '-shortest',  # Match video duration to shorter stream
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to merge audio: {e.stderr}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
    
    def generate_video(self, audio_path: str, output_path: str,
                      start_frame: int = 0, loop_back: bool = True,
                      use_parsing: bool = False,
                      custom_img_dir: Optional[str] = None,
                      custom_lms_dir: Optional[str] = None,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Generate a complete video from audio using standard processing.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output video
            start_frame: Starting frame index
            loop_back: Whether to loop back to start frame
            use_parsing: Whether to use parsing masks
            custom_img_dir: Optional custom frames directory
            custom_lms_dir: Optional custom landmarks directory
            progress_callback: Progress callback(current, total, message)
            
        Returns:
            Path to the generated video file
        """
        # Ensure models are loaded
        if self.unet_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Process audio
        if progress_callback:
            progress_callback(0, 100, "Processing audio...")
        
        audio_feats = self._prepare_audio_features(audio_path)
        total_frames = audio_feats.shape[0]
        
        # Set up directories
        img_dir = Path(custom_img_dir) if custom_img_dir else self.img_dir
        lms_dir = Path(custom_lms_dir) if custom_lms_dir else self.lms_dir
        
        # Check directories exist
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not lms_dir.exists():
            raise ValueError(f"Landmarks directory not found: {lms_dir}")
        
        # Count available frames
        img_files = sorted([f for f in img_dir.glob("*.jpg")])
        num_available_frames = len(img_files)
        
        if num_available_frames == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        # Set up parsing if requested
        parsing_dir = None
        if use_parsing and not custom_img_dir:
            parsing_dir = self.parsing_dir
            if not parsing_dir.exists():
                print(f"Warning: Parsing directory not found: {parsing_dir}")
                parsing_dir = None
        
        # Initialize frame stepping
        generated_frames = []
        step_stride = 1
        img_idx = 0
        
        # Log settings
        print(f"Generating {total_frames} frames at {self.fps} FPS")
        print(f"Available frames: {num_available_frames}")
        print(f"Loop back: {loop_back}")
        
        # Generate frames
        for i in range(total_frames):
            if progress_callback:
                progress_callback(i, total_frames, f"Generating frame {i+1}/{total_frames}")
            
            # Handle frame indexing with looping
            if loop_back:
                # Reverse at midpoint
                if img_idx >= num_available_frames - 1 or img_idx >= total_frames // 2:
                    step_stride = -1
                elif img_idx <= 0:
                    step_stride = 1
            else:
                # Cycle through all frames
                if img_idx >= num_available_frames - 1:
                    img_idx = 0
            
            # Load frame data
            frame_idx = img_idx + start_frame
            img, lms, parsing = self._load_frame_data(
                frame_idx, img_dir, lms_dir, parsing_dir
            )
            
            # Get audio features for this frame
            audio_feat = get_audio_features(audio_feats, i).numpy()
            
            # Generate frame
            result_frame = self._generate_single_frame(img, audio_feat, lms, parsing)
            generated_frames.append(result_frame)
            
            # Update index
            img_idx += step_stride
        
        # Write video
        if progress_callback:
            progress_callback(total_frames, total_frames, "Writing video...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._write_video_file(generated_frames, output_path, audio_path)
        
        print(f"[INFO] Video saved to: {output_path}")
        return output_path
    
    def run_cli(self, audio_path: str, output_dir: str = "./result",
                checkpoint_number: Optional[int] = None,
                start_frame: int = 0, loop_back: bool = True,
                use_parsing: bool = False, asr_mode: str = "ave") -> str:
        """
        Run inference in CLI mode with progress bar.
        
        Args:
            audio_path: Path to input audio
            output_dir: Output directory
            checkpoint_number: Specific checkpoint to load
            start_frame: Starting frame index
            loop_back: Whether to loop back
            use_parsing: Whether to use parsing
            asr_mode: Audio encoder mode
            
        Returns:
            Path to generated video
        """
        # Load models
        checkpoint_file = self.load_models(checkpoint_number, mode=asr_mode)
        checkpoint_name = Path(checkpoint_file).stem
        
        # Generate output filename
        audio_name = Path(audio_path).stem
        output_path = os.path.join(
            output_dir,
            f"{self.model_name}_{audio_name}_{checkpoint_name}.mp4"
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
            start_frame=start_frame,
            loop_back=loop_back,
            use_parsing=use_parsing,
            progress_callback=tqdm_callback
        )