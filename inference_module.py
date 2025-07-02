import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable, Tuple
import time
import tempfile
import shutil
import sys
sys.path.append('./data_utils')

from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
from data_utils.get_landmark import Landmark


class VideoProcessor:
    """Process uploaded videos to extract frames and landmarks."""
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize video processor.
        
        Args:
            temp_dir: Temporary directory for processing. Created if None.
        """
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="synctalk_")
        else:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.landmarks_dir = os.path.join(self.temp_dir, "landmarks")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.landmarks_dir, exist_ok=True)
        
        self.landmark_detector = None
        
    def _init_landmark_detector(self):
        """Initialize landmark detector lazily."""
        if self.landmark_detector is None:
            self.landmark_detector = Landmark()
    
    def extract_frames(self, video_path: str, progress_callback: Optional[Callable] = None) -> int:
        """
        Extract frames from video, converting to 25fps if needed.
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Number of frames extracted
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert to 25fps if needed
        video_to_process = video_path
        if abs(fps - 25.0) > 0.1:  # Not 25fps
            cap.release()
            if progress_callback:
                progress_callback(0, 100, "Converting video to 25fps...")
            
            converted_path = os.path.join(self.temp_dir, "video_25fps.mp4")
            cmd = f'ffmpeg -y -v error -nostats -i "{video_path}" -vf "fps=25" -c:v libx264 "{converted_path}"'
            os.system(cmd)
            video_to_process = converted_path
            cap = cv2.VideoCapture(converted_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        if progress_callback:
            progress_callback(0, total_frames, "Extracting frames...")
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(self.frames_dir, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames, f"Extracting frame {frame_count}/{total_frames}")
        
        cap.release()
        return frame_count
    
    def detect_landmarks(self, progress_callback: Optional[Callable] = None) -> bool:
        """
        Detect landmarks for all extracted frames.
        
        Args:
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            True if successful, False otherwise
        """
        self._init_landmark_detector()
        
        frame_files = sorted(
            [f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )
        
        if not frame_files:
            return False
            
        total_frames = len(frame_files)
        if progress_callback:
            progress_callback(0, total_frames, "Detecting landmarks...")
        
        last_landmarks = None
        successful_detections = 0
        
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(self.frames_dir, frame_file)
            lms_path = os.path.join(self.landmarks_dir, frame_file.replace('.jpg', '.lms'))
            
            # Detect landmarks
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
                        print(f"Warning: No landmarks detected for frame {idx}")
                        
            except Exception as e:
                print(f"Error detecting landmarks for frame {idx}: {e}")
                # Use last valid landmarks if available
                if last_landmarks:
                    with open(lms_path, "w") as f:
                        f.write(last_landmarks)
            
            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(idx + 1, total_frames, f"Detecting landmarks {idx + 1}/{total_frames}")
        
        if progress_callback:
            progress_callback(total_frames, total_frames, "Landmark detection complete")
            
        return successful_detections > 0
    
    def process_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> Tuple[str, str, int]:
        """
        Process video to extract frames and landmarks.
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Tuple of (frames_dir, landmarks_dir, num_frames)
        """
        # Extract frames
        num_frames = self.extract_frames(video_path, progress_callback)
        
        if num_frames == 0:
            raise ValueError("No frames extracted from video")
        
        # Detect landmarks
        success = self.detect_landmarks(progress_callback)
        
        if not success:
            raise ValueError("Failed to detect any landmarks in video")
        
        return self.frames_dir, self.landmarks_dir, num_frames
    
    def cleanup(self):
        """Remove temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


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
                      custom_img_dir: Optional[str] = None,
                      custom_lms_dir: Optional[str] = None,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Generate a complete video from audio.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output video
            start_frame: Starting frame index
            loop_back: Whether to loop back to start frame
            use_parsing: Whether to use parsing masks
            custom_img_dir: Optional custom directory with frames (overrides model dataset)
            custom_lms_dir: Optional custom directory with landmarks (overrides model dataset)
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
        
        # Use custom directories if provided, otherwise use model dataset
        img_dir = custom_img_dir if custom_img_dir else self.img_dir
        lms_dir = custom_lms_dir if custom_lms_dir else self.lms_dir
        
        # Check available frames
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        len_img = len(img_files)
        
        # Get video dimensions from first frame
        exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
        h, w = exm_img.shape[:2]
        
        # Prepare parsing directory if needed
        parsing_dir = None
        if use_parsing and not custom_img_dir:  # Only use parsing for model dataset
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
            img_path = os.path.join(img_dir, f"{img_idx + start_frame}.jpg")
            lms_path = os.path.join(lms_dir, f"{img_idx + start_frame}.lms")
            
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