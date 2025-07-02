import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable, Tuple
import time

from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features


class SyncTalkInference:
    """Modular inference class for SyncTalk 2D video generation."""
    
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize the inference module.
        
        Args:
            model_name: Name of the model (e.g., "AD2.2", "LS1")
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set up paths
        self.checkpoint_path = os.path.join("./checkpoint", model_name)
        self.dataset_dir = os.path.join("./dataset", model_name)
        self.img_dir = os.path.join(self.dataset_dir, "full_body_img/")
        self.lms_dir = os.path.join(self.dataset_dir, "landmarks/")
        
        # Models
        self.audio_encoder = None
        self.unet_model = None
        
        # Video parameters
        self.fps = 25  # Default FPS, will be adjusted based on mode
        
    def load_models(self, checkpoint_number: Optional[int] = None, mode: str = "ave"):
        """
        Load the audio encoder and U-Net models.
        
        Args:
            checkpoint_number: Specific checkpoint number to load. If None, loads the latest.
            mode: Audio feature mode ("ave", "hubert", or "wenet")
        """
        # Load audio encoder
        self.audio_encoder = AudioEncoder().to(self.device).eval()
        ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth', map_location=self.device)
        self.audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
        
        # Load U-Net model
        if checkpoint_number is None:
            # Get the latest checkpoint
            checkpoints = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pth')]
            checkpoint = sorted(checkpoints, key=lambda x: int(x.split(".")[0]))[-1]
            checkpoint_file = os.path.join(self.checkpoint_path, checkpoint)
        else:
            checkpoint_file = os.path.join(self.checkpoint_path, f"{checkpoint_number}.pth")
        
        self.unet_model = Model(6, mode).to(self.device)
        self.unet_model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        self.unet_model.eval()
        
        # Store mode and checkpoint info
        self.mode = mode
        self.checkpoint_file = checkpoint_file
        
        # Adjust FPS based on mode
        if mode == "wenet":
            self.fps = 20
        else:
            self.fps = 25
            
        return checkpoint_file
    
    def process_audio(self, audio_path: str) -> np.ndarray:
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
            
        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)],
                                dim=0).numpy()
        
        return audio_feats
    
    def generate_frame(self, img: np.ndarray, audio_feat: np.ndarray, 
                      lms: np.ndarray, parsing: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a single frame using the U-Net model.
        
        Args:
            img: Input image
            audio_feat: Audio features for this frame
            lms: Facial landmarks
            parsing: Optional parsing image for masked regions
            
        Returns:
            Generated frame
        """
        # Extract face region using landmarks
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        
        # Crop and resize
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img_par = crop_img.copy()
        h, w = crop_img.shape[:2]
        
        if parsing is not None:
            crop_parsing_img = parsing[ymin:ymax, xmin:xmax]
        
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        
        # Create masked input
        img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 310, 305), (0, 0, 0), -1)
        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)
        
        # Convert to tensors
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
        
        # Prepare audio features
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)
        elif self.mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        elif self.mode == "ave":
            audio_feat = audio_feat.reshape(32, 16, 16)
        
        audio_feat_T = torch.from_numpy(audio_feat[None]).to(self.device)
        img_concat_T = img_concat_T.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            pred = self.unet_model(img_concat_T, audio_feat_T)[0]
        
        pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)
        
        # Composite back
        crop_img_ori[4:324, 4:324] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply parsing mask if available
        if parsing is not None:
            parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | \
                          (crop_parsing_img == [255, 255, 255]).all(axis=2)
            crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
        
        # Put back into original image
        result_img = img.copy()
        result_img[ymin:ymax, xmin:xmax] = crop_img_ori
        
        return result_img
    
    def generate_video(self, audio_path: str, output_path: str,
                      start_frame: int = 0, loop_back: bool = True,
                      use_parsing: bool = False,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Generate a complete video from audio.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output video
            start_frame: Starting frame index
            loop_back: Whether to loop back to start frame
            use_parsing: Whether to use parsing masks
            progress_callback: Optional callback for progress updates
                              Called with (current_frame, total_frames, message)
                              
        Returns:
            Path to the generated video file
        """
        # Process audio
        if progress_callback:
            progress_callback(0, 100, "Processing audio...")
        
        audio_feats = self.process_audio(audio_path)
        total_frames = audio_feats.shape[0]
        
        # Check available frames
        len_img = len(os.listdir(self.img_dir)) - 1
        
        # Get video dimensions from first frame
        exm_img = cv2.imread(os.path.join(self.img_dir, "0.jpg"))
        h, w = exm_img.shape[:2]
        
        # Prepare parsing directory if needed
        parsing_dir = None
        if use_parsing:
            parsing_dir = os.path.join(self.dataset_dir, "parsing/")
            if not os.path.exists(parsing_dir):
                print(f"Warning: Parsing directory not found at {parsing_dir}")
                parsing_dir = None
        
        # Create temporary video file
        temp_output = output_path.replace('.mp4', 'temp.mp4')
        video_writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                      self.fps, (w, h))
        
        # Initialize frame stepping
        step_stride = 0
        img_idx = 0
        
        # Log loop settings
        print(f'Frames to be processed: {total_frames}')
        print(f'Loop back enabled: {loop_back}')
        if loop_back:
            turning_point = total_frames // 2
            print(f'Video will reverse at frame {turning_point} (midpoint of {total_frames} audio frames)')
        else:
            print(f'Video will cycle through all {len_img} available frames')
        
        # Generate frames
        for i in range(total_frames):
            # Update progress
            if progress_callback:
                progress_callback(i, total_frames, f"Generating frame {i}/{total_frames}")
            
            # Handle frame indexing with looping logic
            if img_idx > len_img - 1 or (loop_back and img_idx > total_frames / 2):
                step_stride = -1
            if img_idx < 1:
                step_stride = 1
            img_idx += step_stride
            
            # Load current frame and landmarks
            img_path = os.path.join(self.img_dir, f"{img_idx + start_frame}.jpg")
            lms_path = os.path.join(self.lms_dir, f"{img_idx + start_frame}.lms")
            
            img = cv2.imread(img_path)
            
            # Load landmarks
            lms_list = []
            with open(lms_path, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            lms = np.array(lms_list, dtype=np.int32)
            
            # Load parsing if available
            parsing = None
            if parsing_dir:
                parsing_path = os.path.join(parsing_dir, f"{img_idx + start_frame}.png")
                if os.path.exists(parsing_path):
                    parsing = cv2.imread(parsing_path)
            
            # Get audio features for this frame
            audio_feat = get_audio_features(audio_feats, i).numpy()
            
            # Generate frame
            result_frame = self.generate_frame(img, audio_feat, lms, parsing)
            
            # Write frame
            video_writer.write(result_frame)
        
        video_writer.release()
        
        # Merge with audio using ffmpeg
        if progress_callback:
            progress_callback(total_frames, total_frames, "Merging audio...")
        
        ffmpeg_cmd = f"ffmpeg -y -v error -nostats -i {temp_output} -i {audio_path} -c:v libx264 -c:a aac -crf 20 {output_path}"
        os.system(ffmpeg_cmd)
        
        # Clean up temp file
        os.remove(temp_output)
        
        print(f"[INFO] ===== Saved video to {output_path} =====")
        
        return output_path
    
    def run_cli(self, audio_path: str, start_frame: int = 0, 
                loop_back: bool = True, use_parsing: bool = False, 
                asr_mode: str = "ave") -> str:
        """
        Run inference in CLI mode with tqdm progress bar.
        
        Args:
            audio_path: Path to input audio
            start_frame: Starting frame index
            loop_back: Whether to loop back
            use_parsing: Whether to use parsing
            asr_mode: Audio encoder mode
            
        Returns:
            Path to generated video
        """
        # Load models
        checkpoint_file = self.load_models(mode=asr_mode)
        checkpoint_name = os.path.basename(checkpoint_file).split('.')[0]
        
        # Generate output filename
        audio_name = os.path.basename(audio_path).split('.')[0]
        output_path = os.path.join("./result", f"{self.model_name}_{audio_name}_{checkpoint_name}.mp4")
        os.makedirs("./result", exist_ok=True)
        
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