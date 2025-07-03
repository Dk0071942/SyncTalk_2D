import gradio as gr
import os
from pathlib import Path
import time
import traceback
from inference_module import SyncTalkInference, VideoProcessor


class SyncTalkGradio:
    def __init__(self):
        self.output_dir = Path("./gradio_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def discover_models(self):
        """Discover available models in the checkpoint directory"""
        checkpoint_dir = Path("./checkpoint")
        models = []
        
        if checkpoint_dir.exists():
            for model_dir in checkpoint_dir.iterdir():
                if model_dir.is_dir():
                    # Check if there are .pth files
                    pth_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*/*.pth"))
                    if pth_files:
                        models.append(model_dir.name)
        
        return sorted(models)
    
    def generate_video(self, model_name, audio_file, video_file=None, start_frame=0, asr_mode="ave", 
                      loop_back=True, use_core_clips=False, vad_threshold=0.5, 
                      min_silence_duration=0.75, visualize_vad=False, progress=gr.Progress()):
        """Generate video using the new inference module with optional video template"""
        
        try:
            # Validate inputs
            if not model_name:
                return None, "Please select a model"
            
            if not audio_file:
                return None, "Please upload an audio file"
            
            # Update progress
            progress(0.1, desc="Initializing model...")
            
            # Initialize inference module
            inference = SyncTalkInference(model_name)
            
            # Load models
            progress(0.2, desc="Loading models...")
            try:
                checkpoint_file = inference.load_models(mode=asr_mode)
                print(f"Loaded checkpoint: {checkpoint_file}")
            except Exception as e:
                return None, f"Failed to load model: {str(e)}"
            
            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_name = Path(audio_file).stem
            output_filename = f"{model_name}_{audio_name}_{timestamp}.mp4"
            output_path = self.output_dir / output_filename
            
            # Process uploaded video if provided
            custom_img_dir = None
            custom_lms_dir = None
            video_processor = None
            
            if video_file:
                progress(0.25, desc="Processing uploaded video...")
                video_processor = VideoProcessor()
                
                try:
                    # Define progress callback for video processing
                    def video_progress_callback(current, total, message):
                        # Scale progress from 0.25 to 0.3
                        if total > 0:
                            scaled_progress = 0.25 + (current / total) * 0.05
                            progress(scaled_progress, desc=message)
                    
                    custom_img_dir, custom_lms_dir, num_frames = video_processor.process_video(
                        video_file, progress_callback=video_progress_callback
                    )
                    print(f"Processed video: {num_frames} frames extracted")
                    
                except Exception as e:
                    if video_processor:
                        video_processor.cleanup()
                    return None, f"Failed to process video: {str(e)}"
            
            # Define progress callback for Gradio
            def gradio_progress_callback(current, total, message):
                # Scale progress from 0.3 to 0.95
                if total > 0:
                    scaled_progress = 0.3 + (current / total) * 0.65
                    progress(scaled_progress, desc=message)
            
            # Generate video
            progress(0.3, desc="Starting video generation...")
            generated_path = inference.generate_video(
                audio_path=audio_file,
                output_path=str(output_path),
                start_frame=start_frame if not video_file else 0,  # Start from 0 for custom videos
                loop_back=loop_back,
                use_parsing=False,  # Can be exposed as option later
                custom_img_dir=custom_img_dir,
                custom_lms_dir=custom_lms_dir,
                use_core_clips=use_core_clips,
                vad_threshold=vad_threshold,
                min_silence_duration=min_silence_duration,
                visualize_vad=visualize_vad,
                progress_callback=gradio_progress_callback
            )
            
            # Cleanup video processor if used
            if video_processor:
                video_processor.cleanup()
            
            # Finalize
            progress(1.0, desc="Done!")
            
            # Verify file exists
            if os.path.exists(generated_path):
                return generated_path, "Video generated successfully!"
            else:
                return None, f"Video generation completed but file not found at {generated_path}"
                
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg


def create_interface():
    """Create Gradio interface using the new inference module"""
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
                    return gr.Dropdown(choices=models, value=models[0] if models else None)
                
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                refresh_btn.click(refresh_models, outputs=[model_dropdown])
                
                # Audio input
                audio_input = gr.Audio(
                    label="Upload Audio (WAV format)",
                    type="filepath"
                )
                
                # Video input (optional)
                video_input = gr.Video(
                    label="Upload Video Template (Optional) - Leave empty to use model's default character"
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
                    
                    # Core clips options
                    with gr.Group():
                        use_core_clips = gr.Checkbox(
                            value=False,
                            label="Use Core Clips Mode",
                            info="Use pre-recorded clips instead of training dataset"
                        )
                        
                        with gr.Row():
                            vad_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.5,
                                step=0.1,
                                label="VAD Threshold",
                                info="Speech detection sensitivity"
                            )
                            
                            min_silence_duration = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.75,
                                step=0.05,
                                label="Min Silence Duration (s)",
                                info="Minimum duration to keep as silence"
                            )
                        
                        visualize_vad = gr.Checkbox(
                            value=False,
                            label="Save VAD Visualization",
                            info="Save a plot showing speech/silence detection"
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
        
        # Examples
        if available_models:
            gr.Markdown("## Examples")
            # Check for demo audio files
            demo_files = []
            if os.path.exists("demo/talk_hb.wav"):
                demo_files.append("demo/talk_hb.wav")
            if os.path.exists("pause_audio_test_23s.wav"):
                demo_files.append("pause_audio_test_23s.wav")
            
            # Filter examples to only include available models
            example_list = []
            for model in ["LS1", "AD2.2"]:
                if model in available_models and demo_files:
                    # Add example with None for video_input and default values for new params
                    example_list.append([model, demo_files[0], None, 0, "ave", True, False, 0.5, 0.75, False])
            
            if example_list:
                gr.Examples(
                    examples=example_list,
                    inputs=[model_dropdown, audio_input, video_input, start_frame, asr_mode, 
                           loop_back, use_core_clips, vad_threshold, min_silence_duration, visualize_vad],
                    outputs=[output_video, status_text],
                    fn=app.generate_video,
                    cache_examples=False
                )
        
        # Instructions
        gr.Markdown("""
        ## Instructions
        
        1. **Select a Model**: Choose from the available trained models in the dropdown
        2. **Upload Audio**: Upload a WAV audio file (clear speech works best)
        3. **Upload Video** (Optional): Upload a video to use as template character
           - Leave empty to use the model's default character
           - Upload any video with a clear face to use that person as template
        4. **Configure Options** (Optional): Adjust advanced settings if needed
        5. **Generate**: Click the Generate Video button and wait for processing
        6. **Download**: Right-click on the video to save it
        
        ## Notes
        
        - Video generation typically takes 1-3 minutes depending on audio length
        - When uploading a video template, ensure the face is clearly visible
        - Videos will be automatically converted to 25fps if needed
        - The model works best with the same language it was trained on
        - Generated videos are saved in the `gradio_outputs` folder
        - Enable "Loop back to start frame" for seamless video loops
        
        ## Core Clips Mode
        
        - Enable "Use Core Clips Mode" to use pre-recorded body movements
        - This mode uses Voice Activity Detection to select appropriate clips
        - Talking clips are used during speech, silence clips during pauses
        - Adjust VAD Threshold to fine-tune speech detection sensitivity
        - Lower Min Silence Duration to make transitions more responsive
        """)
        
        # Set up event
        generate_btn.click(
            fn=app.generate_video,
            inputs=[model_dropdown, audio_input, video_input, start_frame, asr_mode, 
                   loop_back, use_core_clips, vad_threshold, min_silence_duration, visualize_vad],
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