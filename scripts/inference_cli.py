"""
Command-line interface for SyncTalk 2D using refactored modules.

This script provides a clean CLI interface for video generation using
either standard or core clips mode.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import synctalk module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from refactored synctalk package
from synctalk.processing import StandardVideoProcessor, CoreClipsProcessor
from synctalk.config import get_default_config, apply_preset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SyncTalk 2D - Generate lip-synced videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--name', type=str, required=True,
                       help='Model name (e.g., LS1, AD2.2)')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to input audio file')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'core_clips'],
                       help='Processing mode to use')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./result',
                       help='Output directory for generated videos')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Custom output filename (without extension)')
    
    # Audio options
    parser.add_argument('--asr', type=str, default='ave',
                       choices=['ave', 'hubert', 'wenet'],
                       help='Audio encoder mode')
    
    # Standard mode options
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (standard mode)')
    parser.add_argument('--loop_back', action='store_true', default=True,
                       help='Loop back to start frame (standard mode)')
    parser.add_argument('--no_loop_back', dest='loop_back', action='store_false',
                       help='Disable loop back')
    parser.add_argument('--use_parsing', action='store_true',
                       help='Use face parsing masks (standard mode)')
    
    # Core clips mode options
    parser.add_argument('--vad_threshold', type=float, default=0.5,
                       help='VAD threshold for speech detection (core clips mode)')
    parser.add_argument('--min_silence_duration', type=float, default=0.75,
                       help='Minimum silence duration in seconds (core clips mode)')
    parser.add_argument('--visualize_vad', action='store_true',
                       help='Save VAD visualization (core clips mode)')
    
    # Custom video options
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to custom video template (standard mode only)')
    
    # Quality presets
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'high_quality', 'fast', 'low_memory'],
                       help='Quality preset to apply')
    
    # Other options
    parser.add_argument('--checkpoint', type=int, default=None,
                       help='Specific checkpoint number to load')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)
    
    # Get configuration
    config = get_default_config(args.name)
    if args.preset != 'default':
        apply_preset(config, args.preset)
    
    # Override device if specified
    if args.device:
        config.model.device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    if args.output_name:
        output_filename = f"{args.output_name}.mp4"
    else:
        audio_name = Path(args.audio_path).stem
        mode_suffix = 'core_clips' if args.mode == 'core_clips' else 'standard'
        checkpoint_suffix = f"ckpt{args.checkpoint}" if args.checkpoint else "latest"
        output_filename = f"{args.name}_{audio_name}_{mode_suffix}_{checkpoint_suffix}.mp4"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    try:
        if args.mode == 'core_clips':
            # Use Core Clips Processor
            print(f"Using Core Clips mode for {args.name}")
            processor = CoreClipsProcessor(args.name, config.model.device)
            
            # Process video with CLI mode (includes progress bar)
            generated_path = processor.run_cli(
                audio_path=args.audio_path,
                output_dir=args.output_dir,
                asr_mode=args.asr,
                vad_threshold=args.vad_threshold,
                min_silence_duration=args.min_silence_duration,
                visualize_vad=args.visualize_vad
            )
        else:
            # Use Standard Processor
            print(f"Using Standard mode for {args.name}")
            processor = StandardVideoProcessor(args.name, config.model.device)
            
            # Handle custom video if provided
            custom_img_dir = None
            custom_lms_dir = None
            
            if args.video_path:
                if not os.path.exists(args.video_path):
                    print(f"Error: Video file not found: {args.video_path}")
                    sys.exit(1)
                
                print("Processing custom video template...")
                from synctalk.processing import VideoProcessor
                video_processor = VideoProcessor()
                
                try:
                    custom_img_dir, custom_lms_dir, num_frames = video_processor.process_video(
                        args.video_path
                    )
                    print(f"Extracted {num_frames} frames from custom video")
                except Exception as e:
                    print(f"Error processing video: {e}")
                    video_processor.cleanup()
                    sys.exit(1)
            
            # Process video with CLI mode
            try:
                generated_path = processor.run_cli(
                    audio_path=args.audio_path,
                    output_dir=args.output_dir,
                    checkpoint_number=args.checkpoint,
                    start_frame=args.start_frame,
                    loop_back=args.loop_back,
                    use_parsing=args.use_parsing,
                    asr_mode=args.asr
                )
            finally:
                # Cleanup custom video processor if used
                if args.video_path and 'video_processor' in locals():
                    video_processor.cleanup()
        
        # Print summary
        print("\n" + "="*60)
        print("VIDEO GENERATION COMPLETE")
        print("="*60)
        print(f"Model: {args.name}")
        print(f"Mode: {args.mode}")
        print(f"Audio: {os.path.basename(args.audio_path)}")
        print(f"Output: {generated_path}")
        
        if os.path.exists(generated_path):
            file_size = os.path.getsize(generated_path) / (1024 * 1024)  # MB
            print(f"Size: {file_size:.2f} MB")
            
            # Check for VAD visualization
            if args.mode == 'core_clips' and args.visualize_vad:
                vad_path = generated_path.replace('.mp4', '_vad.png')
                if os.path.exists(vad_path):
                    print(f"VAD visualization: {vad_path}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()