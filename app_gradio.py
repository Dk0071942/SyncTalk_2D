"""
Gradio web interface for SyncTalk 2D using refactored modules.

This is the updated version that uses the new synctalk package structure
with improved UI for clearer mode selection.
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
    
    def check_core_clips_available(self, model_name):
        """Check if core clips are available for the model."""
        if not model_name:
            return False
        core_clips_dir = Path(f"./dataset/{model_name}/core_clips")
        return core_clips_dir.exists() and len(list(core_clips_dir.glob("*.mp4"))) > 0
    
    def generate_video(self, model_name, audio_file, generation_mode, video_file=None, 
                      start_frame=0, asr_mode="ave", loop_back=True, 
                      vad_threshold=0.5, min_silence_duration=0.75, 
                      visualize_vad=False, quality_preset="default", 
                      progress=gr.Progress()):
        """
        Generate video using the selected method.
        
        Args:
            model_name: Selected model name
            audio_file: Path to audio file
            generation_mode: Selected generation mode
            video_file: Optional video template for custom mode
            start_frame: Starting frame index for training data mode
            asr_mode: Audio encoder mode
            loop_back: Whether to loop back
            vad_threshold: VAD threshold for core clips mode
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
                return None, "❌ Please select a model"
            
            if not audio_file:
                return None, "❌ Please upload an audio file"
            
            if generation_mode == "📹 Custom Video Mode" and not video_file:
                return None, "❌ Please upload a video for Custom Video mode"
            
            if generation_mode == "🎞️ Core Clips Mode" and not self.check_core_clips_available(model_name):
                return None, f"❌ Core clips not available for model '{model_name}'. Please run preprocessing first."
            
            # Update progress
            progress(0.1, desc="Initializing...")
            
            # Map display mode to internal mode
            mode_map = {
                "📊 Training Data Mode": "training_data",
                "📹 Custom Video Mode": "custom",
                "🎞️ Core Clips Mode": "core_clips"
            }
            internal_mode = mode_map.get(generation_mode, "training_data")
            
            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_name = Path(audio_file).stem
            output_filename = f"{model_name}_{audio_name}_{internal_mode}_{timestamp}.mp4"
            output_path = self.output_dir / output_filename
            
            # Process based on selected mode
            custom_img_dir = None
            custom_lms_dir = None
            video_processor = None
            
            # Handle custom video processing
            if internal_mode == "custom":
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
                    return None, f"❌ Failed to process video: {str(e)}"
            
            # Define progress callback for main processing
            def main_progress_callback(current, total, message):
                if total > 0:
                    scaled_progress = 0.3 + (current / total) * 0.65
                    progress(scaled_progress, desc=message)
            
            # Generate video based on mode
            progress(0.3, desc="Starting video generation...")
            
            if internal_mode == "core_clips":
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
                # Use Standard Processor for both training data and custom video modes
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
                    start_frame=start_frame if internal_mode == "training_data" else 0,
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
                status_msg = f"✅ Video generated successfully!\n"
                status_msg += f"📁 Size: {file_size:.2f} MB\n"
                status_msg += f"🎬 Mode: {generation_mode}"
                
                # Add VAD visualization note if applicable
                if internal_mode == "core_clips" and visualize_vad:
                    vad_path = generated_path.replace('.mp4', '_vad.png')
                    if os.path.exists(vad_path):
                        status_msg += f"\n📊 VAD visualization saved: {os.path.basename(vad_path)}"
                
                return generated_path, status_msg
            else:
                return None, f"❌ Video generation completed but file not found at {generated_path}"
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg


def create_interface():
    """Create Gradio interface for SyncTalk 2D."""
    app = SyncTalkGradio()
    
    with gr.Blocks(title="SyncTalk 2D - Lip Sync Video Generator", 
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 SyncTalk 2D - Lip Sync Video Generator
        
        Generate high-quality lip-synced videos using trained SyncTalk 2D models.
        Choose from three generation methods based on your needs.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                gr.Markdown("## 🤖 Step 1: Select Model")
                available_models = app.discover_models()
                default_value = available_models[0] if available_models else None
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="Trained Model",
                    value=default_value
                )
                
                # Refresh models button
                def refresh_models():
                    models = app.discover_models()
                    return gr.Dropdown(choices=models, value=models[0] if models else None)
                
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                refresh_btn.click(refresh_models, outputs=[model_dropdown])
                
                # Audio input
                gr.Markdown("## 🎵 Step 2: Upload Audio")
                audio_input = gr.Audio(
                    label="Audio File (WAV format recommended)",
                    type="filepath"
                )
                
                # Generation mode selection
                gr.Markdown("## 🎯 Step 3: Choose Generation Method")
                generation_mode = gr.Radio(
                    choices=[
                        "📊 Training Data Mode",
                        "📹 Custom Video Mode",
                        "🎞️ Core Clips Mode"
                    ],
                    value="📊 Training Data Mode",
                    label="Generation Method"
                )
                
                # Mode explanations
                with gr.Accordion("ℹ️ Method Explanations", open=True):
                    gr.Markdown("""
                    **📊 Training Data Mode**: Uses frames from the original training dataset. 
                    Fast and reliable with consistent quality.
                    
                    **📹 Custom Video Mode**: Process your own video to match a specific person or style. 
                    Requires a video with clear frontal face views.
                    
                    **🎞️ Core Clips Mode**: Intelligently selects from pre-recorded clips using Voice Activity Detection. 
                    Preserves natural pauses and breathing for realistic results.
                    """)
                
                # Mode-specific settings
                gr.Markdown("## ⚙️ Step 4: Configure Settings")
                
                # Training Data Settings
                with gr.Accordion("📊 Training Data Settings", open=True):
                    gr.Markdown("*Only applies when using Training Data Mode*")
                    start_frame = gr.Number(
                        label="Start Frame (0-indexed)",
                        value=0,
                        precision=0
                    )
                    
                    loop_back = gr.Checkbox(
                        value=True,
                        label="Loop Back (Reverse at midpoint)"
                    )
                
                # Custom Video Settings
                with gr.Accordion("📹 Custom Video Settings", open=True):
                    gr.Markdown("*Only applies when using Custom Video Mode*")
                    video_input = gr.Video(
                        label="Upload Video Template (Clear frontal face required)"
                    )
                
                # Core Clips Settings
                with gr.Accordion("🎞️ Core Clips Settings", open=True):
                    gr.Markdown("*Only applies when using Core Clips Mode*")
                    
                    vad_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        step=0.1,
                        value=0.5,
                        label="VAD Threshold (lower = more sensitive)"
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
                        label="Save VAD Visualization (Speech/silence plot)"
                    )
                
                # Common settings
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    asr_mode = gr.Radio(
                        choices=["ave", "hubert", "wenet"],
                        value="ave",
                        label="Audio Encoder (AVE is default)"
                    )
                    
                    quality_preset = gr.Radio(
                        choices=["default", "high_quality", "fast", "low_memory"],
                        value="high_quality",
                        label="Quality Preset (⚖️ Balanced | 🎨 High Quality | ⚡ Fast | 💾 Low Memory)"
                    )
                
                # Generate button
                gr.Markdown("## 🎬 Step 5: Generate")
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
                
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    lines=4,
                    max_lines=6,
                    interactive=False
                )
            
            with gr.Column(scale=2):
                # Video output
                gr.Markdown("## 📺 Generated Video")
                video_output = gr.Video(
                    label="Output",
                    autoplay=True
                )
                
                # Tips and information
                with gr.Row():
                    with gr.Accordion("💡 Tips & Best Practices", open=False):
                        gr.Markdown("""
                        ### Audio Preparation
                        - Use clean, noise-free audio
                        - WAV format recommended
                        - Normalize audio levels
                        
                        ### Video Requirements (Custom Mode)
                        - Clear frontal face view
                        - Stable camera position
                        - Good lighting
                        - Minimum resolution: 256x256
                        
                        ### Performance Tips
                        - Use "Fast" preset for quick previews
                        - "Low Memory" preset for long videos
                        - Process videos in segments for best results
                        """)
                    
                    with gr.Accordion("📊 Method Comparison", open=False):
                        gr.Markdown("""
                        | Method | Speed | Quality | Flexibility | Best For |
                        |--------|-------|---------|-------------|----------|
                        | **Training Data** | ⚡ Fastest | ⭐⭐⭐ Consistent | ⭐ Limited | Quick generation, batch processing |
                        | **Custom Video** | ⭐⭐ Moderate | ⭐⭐ Variable | ⭐⭐⭐ High | Specific person/style matching |
                        | **Core Clips** | ⚡ Fast | ⭐⭐⭐ High | ⭐⭐ Moderate | Natural pauses, presentations |
                        """)
                
                # Examples
                gr.Markdown("## 📚 Examples")
                gr.Examples(
                    examples=[
                        ["LS1", "demo/talk_hb.wav", "📊 Training Data Mode", None, 0, "ave", True, 0.5, 0.75, False, "high_quality"],
                        ["AD2.2", "demo/AD2_Audio.mp3", "🎞️ Core Clips Mode", None, 0, "ave", True, 0.5, 0.75, True, "high_quality"],
                    ],
                    inputs=[model_dropdown, audio_input, generation_mode, video_input,
                           start_frame, asr_mode, loop_back, vad_threshold,
                           min_silence_duration, visualize_vad, quality_preset],
                    outputs=[video_output, status_output],
                    fn=app.generate_video,
                    cache_examples=False
                )
        
        # Wire up the generate button
        generate_btn.click(
            fn=app.generate_video,
            inputs=[
                model_dropdown, audio_input, generation_mode, video_input,
                start_frame, asr_mode, loop_back, vad_threshold,
                min_silence_duration, visualize_vad, quality_preset
            ],
            outputs=[video_output, status_output]
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