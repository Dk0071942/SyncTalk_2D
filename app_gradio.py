import gradio as gr
import os
from pathlib import Path
import time
import traceback
from inference_module import SyncTalkInference


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
    
    def generate_video(self, model_name, audio_file, start_frame=0, asr_mode="ave", 
                      loop_back=True, progress=gr.Progress()):
        """Generate video using the new inference module"""
        
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
                start_frame=start_frame,
                loop_back=loop_back,
                use_parsing=False,  # Can be exposed as option later
                progress_callback=gradio_progress_callback
            )
            
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
                    example_list.append([model, demo_files[0], 0, "ave", True])
            
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
        - Enable "Loop back to start frame" for seamless video loops
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