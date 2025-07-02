import gradio as gr
import os
import torch
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
import time
import traceback
import subprocess


class SyncTalkGradio:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_encoder = None
        self.current_model = None
        self.current_model_name = None
        
    def discover_models(self):
        """Discover available models in the checkpoint directory"""
        checkpoint_dir = Path("./checkpoint")
        models = []
        
        if checkpoint_dir.exists():
            for model_dir in checkpoint_dir.iterdir():
                if model_dir.is_dir():
                    # Check if there are .pth files in the directory
                    pth_files = list(model_dir.glob("*.pth"))
                    if pth_files:
                        # For nested directories like LS1/LS1/
                        if (model_dir / model_dir.name).exists():
                            nested_pth_files = list((model_dir / model_dir.name).glob("*.pth"))
                            if nested_pth_files:
                                models.append(model_dir.name)
                        else:
                            models.append(model_dir.name)
        
        return sorted(models)
    
    def get_latest_checkpoint(self, model_name):
        """Get the latest checkpoint for a given model"""
        checkpoint_path = Path("./checkpoint") / model_name
        
        # Check for nested directory structure
        if (checkpoint_path / model_name).exists():
            checkpoint_path = checkpoint_path / model_name
        
        # Get all .pth files and sort by number
        pth_files = list(checkpoint_path.glob("*.pth"))
        if not pth_files:
            raise ValueError(f"No checkpoint files found for model {model_name}")
        
        # Sort by the numeric part of the filename
        pth_files.sort(key=lambda x: int(x.stem))
        return str(pth_files[-1])
    
    def load_model(self, model_name, asr_mode="ave"):
        """Load a specific model"""
        if self.current_model_name == model_name:
            return True
        
        try:
            checkpoint_path = self.get_latest_checkpoint(model_name)
            print(f"Loading model from {checkpoint_path}")
            
            # Load the model
            self.current_model = Model(6, mode=asr_mode).to(self.device)
            self.current_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.current_model.eval()
            self.current_model_name = model_name
            
            # Load audio encoder if not already loaded
            if self.audio_encoder is None and asr_mode == "ave":
                self.audio_encoder = AudioEncoder().to(self.device).eval()
                ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth', map_location=self.device)
                self.audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def process_audio_features(self, audio_path, asr_mode="ave"):
        """Process audio file and extract features"""
        if asr_mode == "ave":
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
        else:
            raise NotImplementedError(f"ASR mode {asr_mode} not implemented in Gradio interface yet")
    
    def generate_video(self, model_name, audio_file, start_frame=0, asr_mode="ave", loop_back=True, progress=gr.Progress()):
        """Generate lip-synced video from audio"""
        try:
            # Validate inputs
            if not model_name:
                return None, "Please select a model"
            
            if not audio_file:
                return None, "Please upload an audio file"
            
            # Load model
            progress(0.1, desc="Loading model...")
            if not self.load_model(model_name, asr_mode):
                return None, f"Failed to load model {model_name}"
            
            # Process audio
            progress(0.2, desc="Processing audio...")
            audio_feats = self.process_audio_features(audio_file, asr_mode)
            
            # Get dataset directory and check if it exists
            dataset_dir = Path("./dataset") / model_name
            if not dataset_dir.exists():
                return None, f"Dataset directory not found for model {model_name}"
            
            img_dir = dataset_dir / "full_body_img"
            lms_dir = dataset_dir / "landmarks"
            
            # Check if required directories exist
            if not img_dir.exists() or not lms_dir.exists():
                return None, f"Required data directories not found for model {model_name}"
            
            # Get example image for dimensions
            len_img = len(list(img_dir.glob("*.jpg"))) - 1
            exm_img = cv2.imread(str(img_dir / "0.jpg"))
            if exm_img is None:
                return None, "Failed to load reference image"
            
            h, w = exm_img.shape[:2]
            
            # Create temporary output file
            temp_dir = tempfile.mkdtemp()
            temp_video = os.path.join(temp_dir, "output_temp.mp4")
            final_video = os.path.join(temp_dir, "output_final.mp4")
            
            # Initialize video writer - use mp4v codec for MP4 format
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video, fourcc, 25, (w, h))
            
            # Generate frames with improved cycling logic
            progress(0.3, desc="Generating video frames...")
            
            total_frames = audio_feats.shape[0]
            available_frames = len_img  # Total number of frames in the dataset
            
            # Create frame sequence that starts and ends with the same frame
            frame_sequence = []
            
            if total_frames <= available_frames:
                # Audio is shorter than video: use frames from start_frame forward
                for i in range(total_frames):
                    frame_sequence.append((start_frame + i) % available_frames)
            else:
                # Audio is longer than video: create a looping pattern
                # Create a proper forward-backward cycle
                # Forward: 0, 1, 2, ..., available_frames-1
                # Backward: available_frames-2, ..., 1 (excluding 0 and available_frames-1 to avoid repetition)
                forward_indices = list(range(available_frames))
                backward_indices = list(range(available_frames - 2, 0, -1))
                
                # Full cycle of indices
                full_cycle = forward_indices + backward_indices
                cycle_length = len(full_cycle)
                
                # Fill the sequence
                for i in range(total_frames):
                    # Get the index in the cycle
                    cycle_pos = i % cycle_length
                    # Get the actual frame number (offset by start_frame)
                    frame_idx = (start_frame + full_cycle[cycle_pos]) % available_frames
                    frame_sequence.append(frame_idx)
            
            # Ensure we end with the starting frame for seamless loop (if enabled)
            if loop_back and total_frames > 1 and frame_sequence[-1] != start_frame:
                # Adjust the last few frames to smoothly return to start
                transition_frames = min(15, total_frames // 10)  # Use up to 15 frames for transition
                
                # Force the last frames to be the start frame
                # This ensures we always end exactly where we started
                for i in range(transition_frames):
                    idx = total_frames - transition_frames + i
                    if idx < total_frames and idx >= 0:
                        # For the last half of the transition, use start_frame
                        if i >= transition_frames // 2:
                            frame_sequence[idx] = start_frame
            
            # Process each frame
            for i in range(total_frames):
                # Update progress
                progress(0.3 + 0.5 * (i / total_frames), desc=f"Generating frame {i+1}/{total_frames}...")
                
                # Get the frame index for this position
                img_idx = frame_sequence[i]
                
                # Load image and landmarks
                img_path = img_dir / f"{img_idx}.jpg"
                lms_path = lms_dir / f"{img_idx}.lms"
                
                if not img_path.exists() or not lms_path.exists():
                    print(f"Warning: Missing files for frame {img_idx}")
                    continue
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Read landmarks
                lms_list = []
                with open(lms_path, "r") as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        arr = line.split(" ")
                        arr = np.array(arr, dtype=np.float32)
                        lms_list.append(arr)
                
                lms = np.array(lms_list, dtype=np.int32)
                
                # Crop face region
                xmin = lms[1][0]
                ymin = lms[52][1]
                xmax = lms[31][0]
                width = xmax - xmin
                ymax = ymin + width
                
                crop_img = img[ymin:ymax, xmin:xmax]
                crop_h, crop_w = crop_img.shape[:2]
                
                # Resize and prepare input
                crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
                crop_img_ori = crop_img.copy()
                img_real_ex = crop_img[4:324, 4:324].copy()
                img_real_ex_ori = img_real_ex.copy()
                
                # Create masked input
                img_masked = cv2.rectangle(img_real_ex_ori.copy(), (5,5,310,305), (0,0,0), -1)
                img_masked = img_masked.transpose(2,0,1).astype(np.float32)
                img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
                
                img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
                img_masked_T = torch.from_numpy(img_masked / 255.0)
                img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None]
                
                # Get audio features
                audio_feat = get_audio_features(audio_feats, i)
                if asr_mode == "ave":
                    audio_feat = audio_feat.reshape(32, 16, 16)
                elif asr_mode == "hubert":
                    audio_feat = audio_feat.reshape(32, 32, 32)
                elif asr_mode == "wenet":
                    audio_feat = audio_feat.reshape(256, 16, 32)
                
                audio_feat = audio_feat[None]
                if isinstance(audio_feat, np.ndarray):
                    audio_feat = torch.from_numpy(audio_feat).to(self.device)
                else:
                    audio_feat = audio_feat.to(self.device)
                img_concat_T = img_concat_T.to(self.device)
                
                # Generate prediction
                with torch.no_grad():
                    pred = self.current_model(img_concat_T, audio_feat)[0]
                
                pred = pred.cpu().numpy().transpose(1,2,0) * 255
                pred = np.array(pred, dtype=np.uint8)
                
                # Blend prediction back
                crop_img_ori[4:324, 4:324] = pred
                crop_img_ori = cv2.resize(crop_img_ori, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
                img[ymin:ymax, xmin:xmax] = crop_img_ori
                
                video_writer.write(img)
            
            video_writer.release()
            
            # Merge with audio
            progress(0.9, desc="Adding audio...")
            cmd = f"ffmpeg -y -v error -nostats -i {temp_video} -i {audio_file} -c:v libx264 -c:a aac -crf 20 {final_video}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Copy to persistent location
            output_dir = Path("./gradio_outputs")
            output_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{model_name}_{timestamp}.mp4"
            shutil.copy(final_video, output_path)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            progress(1.0, desc="Done!")
            return str(output_path), "Video generated successfully!"
            
        except Exception as e:
            error_msg = f"Error generating video: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg


def create_interface():
    """Create Gradio interface"""
    app = SyncTalkGradio()
    
    with gr.Blocks(title="SyncTalk 2D - Lip Sync Video Generator") as demo:
        gr.Markdown("""
        # SyncTalk 2D - Lip Sync Video Generator
        
        Generate high-quality lip-synced videos using trained SyncTalk 2D models.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                available_models = app.discover_models()
                default_value = available_models[0] if available_models else None
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="Select Model",
                    value=default_value
                )
                
                # Refresh models button
                def refresh_models():
                    models = app.discover_models()
                    return gr.Dropdown(choices=models)
                
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                refresh_btn.click(refresh_models, outputs=[model_dropdown])
                
                # Audio input
                audio_input = gr.Audio(
                    label="Upload Audio (WAV format)",
                    type="filepath"
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    start_frame = gr.Number(
                        label="Start Frame",
                        value=0,
                        precision=0
                    )
                    
                    asr_mode = gr.Radio(
                        choices=["ave", "hubert", "wenet"],
                        value="ave",
                        label="Audio Encoder"
                    )
                    
                    loop_back = gr.Checkbox(
                        value=True,
                        label="Loop back to start frame"
                    )
                
                # Generate button
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                # Output video
                output_video = gr.Video(
                    label="Generated Video",
                    autoplay=True
                )
                
                # Status message
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3
                )
        
        # Examples - only show if we have models available
        if available_models:
            gr.Markdown("## Examples")
            # Filter examples to only include available models
            example_list = []
            for model in ["LS1", "AD2.2"]:
                if model in available_models:
                    example_list.append([model, "demo/talk_hb.wav", 0, "ave", True])
            
            if example_list:
                gr.Examples(
                    examples=example_list,
                    inputs=[model_dropdown, audio_input, start_frame, asr_mode, loop_back],
                    outputs=[output_video, status_text],
                    fn=app.generate_video,
                    cache_examples=False
                )
        
        # Instructions
        gr.Markdown("""
        ## Instructions
        
        1. **Select a Model**: Choose from the available trained models in the dropdown
        2. **Upload Audio**: Upload a WAV audio file (clear speech works best)
        3. **Configure Options** (Optional): Adjust advanced settings if needed
        4. **Generate**: Click the Generate Video button and wait for processing
        5. **Download**: Right-click on the video to save it
        
        ## Notes
        
        - Video generation typically takes 1-3 minutes depending on audio length
        - Ensure your audio is clear and without background noise
        - The model works best with the same language it was trained on
        - Generated videos are saved in the `gradio_outputs` folder
        """)
        
        # Set up event
        generate_btn.click(
            fn=app.generate_video,
            inputs=[model_dropdown, audio_input, start_frame, asr_mode, loop_back],
            outputs=[output_video, status_text]
        )
    
    return demo


if __name__ == "__main__":
    # Create output directory
    Path("./gradio_outputs").mkdir(exist_ok=True)
    
    # Launch the app
    demo = create_interface()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )