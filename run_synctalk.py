#!/usr/bin/env python3
"""
Main entry point for SyncTalk 2D.

This script provides a unified interface for all SyncTalk functionality,
including both CLI and web interfaces.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description='SyncTalk 2D - Lip Sync Video Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate video from audio')
    generate_parser.add_argument('--name', type=str, required=True,
                                help='Model name (e.g., LS1, AD2.2)')
    generate_parser.add_argument('--audio', type=str, required=True,
                                help='Path to input audio file')
    generate_parser.add_argument('--mode', type=str, default='standard',
                                choices=['standard', 'core_clips'],
                                help='Processing mode')
    generate_parser.add_argument('--output', type=str, default=None,
                                help='Output path (default: auto-generated)')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=7860,
                           help='Port for web server')
    web_parser.add_argument('--share', action='store_true',
                           help='Create public share link')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--name', type=str, required=True,
                            help='Model name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'generate':
        # Import and run inference
        from scripts.inference_cli import main as run_inference
        
        # Build command line arguments
        cmd_args = [
            '--name', args.name,
            '--audio_path', args.audio,
            '--mode', args.mode
        ]
        
        if args.output:
            output_dir = os.path.dirname(args.output)
            output_name = os.path.splitext(os.path.basename(args.output))[0]
            if output_dir:
                cmd_args.extend(['--output_dir', output_dir])
            cmd_args.extend(['--output_name', output_name])
        
        # Replace sys.argv for the inference script
        sys.argv = ['inference_cli.py'] + cmd_args
        run_inference()
        
    elif args.command == 'web':
        # Import and run Gradio app
        from app_gradio import create_interface
        
        print(f"Starting web interface on port {args.port}...")
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        
    elif args.command == 'info':
        # Show model information
        from synctalk.config import load_model_config
        from synctalk.processing import StandardVideoProcessor, CoreClipsProcessor
        
        print(f"\nModel Information: {args.name}")
        print("="*50)
        
        # Load configuration
        config = load_model_config(args.name)
        
        # Check paths
        checkpoint_path = config.model.checkpoint_path
        dataset_path = config.model.dataset_path
        
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"  Exists: {os.path.exists(checkpoint_path)}")
        
        print(f"Dataset path: {dataset_path}")
        print(f"  Exists: {os.path.exists(dataset_path)}")
        
        # Check for core clips
        core_clips_path = f"./core_clips/{args.name}"
        print(f"Core clips path: {core_clips_path}")
        print(f"  Exists: {os.path.exists(core_clips_path)}")
        
        if os.path.exists(core_clips_path):
            processor = CoreClipsProcessor(args.name)
            stats = processor.get_clip_statistics()
            print(f"  Talk clips: {stats['total_talk_clips']}")
            print(f"  Silence clips: {stats['total_silence_clips']}")
        
    elif args.command == 'list':
        # List available models
        from pathlib import Path
        
        print("\nAvailable Models:")
        print("="*50)
        
        checkpoint_dir = Path("./checkpoint")
        if checkpoint_dir.exists():
            models = []
            for model_dir in checkpoint_dir.iterdir():
                if model_dir.is_dir():
                    pth_files = list(model_dir.glob("*.pth"))
                    if pth_files:
                        models.append({
                            'name': model_dir.name,
                            'checkpoints': len(pth_files),
                            'latest': max(pth_files, key=lambda x: x.stat().st_mtime).name
                        })
            
            if models:
                for model in sorted(models, key=lambda x: x['name']):
                    print(f"\n{model['name']}:")
                    print(f"  Checkpoints: {model['checkpoints']}")
                    print(f"  Latest: {model['latest']}")
                    
                    # Check for dataset
                    dataset_path = f"./dataset/{model['name']}"
                    if os.path.exists(dataset_path):
                        print(f"  Dataset: ✓")
                    else:
                        print(f"  Dataset: ✗")
                    
                    # Check for core clips
                    core_clips_path = f"./core_clips/{model['name']}"
                    if os.path.exists(core_clips_path):
                        print(f"  Core clips: ✓")
                    else:
                        print(f"  Core clips: ✗")
            else:
                print("No models found")
        else:
            print("Checkpoint directory not found")


if __name__ == "__main__":
    main()