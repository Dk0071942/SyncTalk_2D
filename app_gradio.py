"""
Gradio web interface for SyncTalk 2D using refactored modules.

This is the updated version that uses the new synctalk package structure.
"""

import gradio as gr
import os
from pathlib import Path
import time
import traceback

# Import from refactored synctalk package
from synctalk.processing import VideoProcessor, StandardVideoProcessor, CoreClipsProcessor
from synctalk.config import get_default_config, apply_preset


class SyncTalkGradio:
    """Gradio application for SyncTalk 2D."""
    
    def __init__(self):
        self.output_dir = Path("./gradio_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def discover_models(self):
        """Discover available models in the checkpoint directory."""
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
    
    def generate_video(self, model_name, audio_file, video_file=None, 
                      start_frame=0, asr_mode="ave", loop_back=True, 
                      use_core_clips=False, vad_threshold=0.5, 
                      min_silence_duration=0.75, visualize_vad=False,
                      quality_preset="default", progress=gr.Progress()):
        """
        Generate video using the refactored processors.
        
        Args:
            model_name: Selected model name
            audio_file: Path to audio file
            video_file: Optional video template
            start_frame: Starting frame index
            asr_mode: Audio encoder mode
            loop_back: Whether to loop back
            use_core_clips: Whether to use core clips mode
            vad_threshold: VAD threshold
            min_silence_duration: Minimum silence duration
            visualize_vad: Whether to visualize VAD
            quality_preset: Quality preset to apply
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (video_path, status_message)
        """
        try:
            # Validate inputs
            if not model_name:
                return None, "Please select a model"
            
            if not audio_file:
                return None, "Please upload an audio file"
            
            # Update progress
            progress(0.1, desc="Initializing...")
            
            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_name = Path(audio_file).stem
            mode_suffix = "core_clips" if use_core_clips else "standard"
            output_filename = f"{model_name}_{audio_name}_{mode_suffix}_{timestamp}.mp4"
            output_path = self.output_dir / output_filename
            
            # Process uploaded video if provided
            custom_img_dir = None
            custom_lms_dir = None
            video_processor = None
            
            if video_file and not use_core_clips:
                progress(0.2, desc="Processing uploaded video...")
                video_processor = VideoProcessor()
                
                try:
                    # Define progress callback for video processing
                    def video_progress_callback(current, total, message):
                        if total > 0:
                            scaled_progress = 0.2 + (current / total) * 0.1
                            progress(scaled_progress, desc=message)
                    
                    custom_img_dir, custom_lms_dir, num_frames = video_processor.process_video(
                        video_file, progress_callback=video_progress_callback
                    )
                    print(f"Processed video: {num_frames} frames extracted")
                    
                except Exception as e:
                    if video_processor:
                        video_processor.cleanup()
                    return None, f"Failed to process video: {str(e)}"
            
            # Define progress callback for main processing
            def main_progress_callback(current, total, message):
                if total > 0:
                    scaled_progress = 0.3 + (current / total) * 0.65
                    progress(scaled_progress, desc=message)
            
            # Generate video based on mode
            progress(0.3, desc="Starting video generation...")
            
            if use_core_clips:
                # Use Core Clips Processor
                processor = CoreClipsProcessor(model_name)
                
                # Get configuration
                config = get_default_config(model_name)
                if quality_preset != "default":
                    apply_preset(config, quality_preset)
                
                # Load models
                progress(0.35, desc="Loading models...")
                generated_path = processor.generate_video(
                    audio_path=audio_file,
                    output_path=str(output_path),
                    asr_mode=asr_mode,
                    vad_threshold=vad_threshold,
                    min_silence_duration=min_silence_duration,
                    visualize_vad=visualize_vad,
                    progress_callback=main_progress_callback
                )
            else:
                # Use Standard Processor
                processor = StandardVideoProcessor(model_name)
                
                # Get configuration
                config = get_default_config(model_name)
                if quality_preset != "default":
                    apply_preset(config, quality_preset)
                
                # Load models
                progress(0.35, desc="Loading models...")
                processor.load_models(mode=asr_mode)
                
                generated_path = processor.generate_video(
                    audio_path=audio_file,
                    output_path=str(output_path),
                    start_frame=start_frame if not video_file else 0,
                    loop_back=loop_back,
                    use_parsing=False,
                    custom_img_dir=custom_img_dir,
                    custom_lms_dir=custom_lms_dir,
                    progress_callback=main_progress_callback
                )
            
            # Cleanup video processor if used
            if video_processor:
                video_processor.cleanup()
            
            # Finalize
            progress(1.0, desc="Done!")
            
            # Verify file exists
            if os.path.exists(generated_path):
                # Get file info
                file_size = os.path.getsize(generated_path) / (1024 * 1024)  # MB
                status_msg = f"Video generated successfully! Size: {file_size:.2f} MB"
                
                # Add VAD visualization note if applicable
                if use_core_clips and visualize_vad:
                    vad_path = generated_path.replace('.mp4', '_vad.png')
                    if os.path.exists(vad_path):
                        status_msg += f"\nVAD visualization saved: {os.path.basename(vad_path)}"
                
                return generated_path, status_msg
            else:
                return None, f"Video generation completed but file not found at {generated_path}"
                
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg


def create_interface():
    """Create Gradio interface for SyncTalk 2D."""
    app = SyncTalkGradio()
    
    with gr.Blocks(title="SyncTalk 2D - Lip Sync Video Generator") as demo:
        gr.Markdown("""
        # SyncTalk 2D - Lip Sync Video Generator
        
        Generate high-quality lip-synced videos using trained SyncTalk 2D models.
        
        **New Features:**
        - 🎯 Standard Mode: Uses training dataset frames
        - 🎬 Core Clips Mode: Uses pre-recorded video segments with VAD
        - 📹 Custom Video Support: Use your own video as template
        - ⚡ Quality Presets: Optimize for quality or speed
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
                    label="Upload Video Template (Optional) - Only for Standard Mode"
                )
                
                # Mode selection
                with gr.Group():
                    use_core_clips = gr.Checkbox(
                        value=False,
                        label="Use Core Clips Mode",
                        info="Use pre-recorded clips with VAD-based selection"
                    )
                
                # Quality preset
                quality_preset = gr.Radio(
                    choices=["default", "high_quality", "fast", "low_memory"],
                    value="default",
                    label="Quality Preset"
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Tab("Standard Mode"):
                        start_frame = gr.Number(
                            label="Start Frame",
                            value=0,
                            precision=0
                        )
                        
                        loop_back = gr.Checkbox(
                            value=True,
                            label="Loop back to start frame"
                        )
                    
                    with gr.Tab("Core Clips Mode"):
                        vad_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            step=0.1,
                            value=0.5,
                            label="VAD Threshold",
                            info="Lower = more sensitive"
                        )
                        
                        min_silence_duration = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=0.75,
                            label="Minimum Silence Duration (seconds)"
                        )
                        
                        visualize_vad = gr.Checkbox(
                            value=False,
                            label="Save VAD Visualization",
                            info="Creates a plot showing speech/silence detection"
                        )
                    
                    with gr.Tab("Audio Settings"):
                        asr_mode = gr.Radio(
                            choices=["ave", "hubert", "wenet"],
                            value="ave",
                            label="Audio Encoder"
                        )
                
                # Generate button
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                
                # Status output
                status_output = gr.Textbox(label="Status", lines=3)
            
            with gr.Column(scale=2):
                # Video output
                video_output = gr.Video(label="Generated Video")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["LS1", "examples/audio1.wav", None, 0, "ave", True, False, 0.5, 0.75, False, "default"],
                        ["AD2.2", "examples/audio2.wav", None, 0, "ave", True, True, 0.5, 0.75, True, "default"],
                    ],
                    inputs=[model_dropdown, audio_input, video_input, start_frame, 
                           asr_mode, loop_back, use_core_clips, vad_threshold, 
                           min_silence_duration, visualize_vad, quality_preset],
                    outputs=[video_output, status_output],
                    fn=app.generate_video,
                    cache_examples=False
                )
        
        # Wire up the interface
        generate_btn.click(
            fn=app.generate_video,
            inputs=[
                model_dropdown, audio_input, video_input, start_frame,
                asr_mode, loop_back, use_core_clips, vad_threshold,
                min_silence_duration, visualize_vad, quality_preset
            ],
            outputs=[video_output, status_output]
        )
        
        # Update UI based on mode selection
        def update_ui_for_mode(use_core_clips):
            return {
                video_input: gr.update(visible=not use_core_clips),
                start_frame: gr.update(visible=not use_core_clips),
                loop_back: gr.update(visible=not use_core_clips),
                vad_threshold: gr.update(visible=use_core_clips),
                min_silence_duration: gr.update(visible=use_core_clips),
                visualize_vad: gr.update(visible=use_core_clips)
            }
        
        use_core_clips.change(
            fn=update_ui_for_mode,
            inputs=[use_core_clips],
            outputs=[video_input, start_frame, loop_back, 
                    vad_threshold, min_silence_duration, visualize_vad]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )