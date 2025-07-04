#!/usr/bin/env python3
"""
Batch preprocessing script for multiple videos
Useful for preprocessing multiple training videos at once
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synctalk.utils.batch_processing import BatchProcessor, create_batch_config_template
from synctalk.utils.preprocessing_utils import check_preprocessing_status


def main():
    parser = argparse.ArgumentParser(
        description='Batch preprocess multiple videos for SyncTalk_2D training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in a directory
  python batch_preprocess.py --video_dir ./videos/
  
  # Process videos from a text list
  python batch_preprocess.py --video_list video_list.txt
  
  # Process videos from JSON config with parallel workers
  python batch_preprocess.py --json_config batch_config.json --workers 4
  
  # Create a template configuration file
  python batch_preprocess.py --create_template
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video_list', type=str,
                           help='Text file with video paths and names (one per line, space-separated)')
    input_group.add_argument('--video_dir', type=str,
                           help='Directory containing video files to process')
    input_group.add_argument('--json_config', type=str,
                           help='JSON file with preprocessing configuration')
    input_group.add_argument('--create_template', action='store_true',
                           help='Create a template batch configuration file')
    
    # Common options
    parser.add_argument('--asr_model', type=str, default='ave', choices=['ave', 'hubert', 'wenet'],
                        help='ASR model to use for audio feature extraction (default: ave)')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Base directory for datasets (default: dataset)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that have already been preprocessed')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--log_file', type=str,
                        help='Save processing log to file')
    parser.add_argument('--results_file', type=str, default='batch_results.json',
                        help='Save results to JSON file (default: batch_results.json)')
    
    args = parser.parse_args()
    
    # Handle template creation
    if args.create_template:
        create_batch_config_template('batch_config_template.json')
        return
    
    # Require input source
    if not any([args.video_list, args.video_dir, args.json_config]):
        parser.error('One of --video_list, --video_dir, or --json_config is required')
    
    # Initialize batch processor
    processor = BatchProcessor(max_workers=args.workers, log_file=args.log_file)
    
    # Collect tasks to process
    tasks = []
    
    if args.video_list:
        # Read from text file
        with open(args.video_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        video_path, name = parts
                        tasks.append({
                            'video_path': video_path,
                            'name': name,
                            'asr_model': args.asr_model
                        })
                    elif len(parts) == 1 and parts[0]:
                        # Use filename as name if not specified
                        video_path = parts[0]
                        name = Path(video_path).stem
                        tasks.append({
                            'video_path': video_path,
                            'name': name,
                            'asr_model': args.asr_model
                        })
                        
    elif args.video_dir:
        # Process all videos in directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_dir = Path(args.video_dir)
        
        if not video_dir.exists():
            print(f"Error: Directory not found: {args.video_dir}")
            sys.exit(1)
        
        for video_file in sorted(video_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                name = video_file.stem  # Use filename without extension as name
                tasks.append({
                    'video_path': str(video_file),
                    'name': name,
                    'asr_model': args.asr_model
                })
                
    elif args.json_config:
        # Use batch processor's config loader
        tasks = processor.load_batch_config(args.json_config)
        # Apply default ASR model if not specified in config
        for task in tasks:
            if 'asr_model' not in task:
                task['asr_model'] = args.asr_model
    
    if not tasks:
        print("No videos found to process!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} videos to process")
    
    # Filter out existing if requested
    if args.skip_existing:
        filtered_tasks = []
        skipped = 0
        for task in tasks:
            dataset_path = os.path.join(args.dataset_dir, task['name'])
            status = check_preprocessing_status(dataset_path)
            if status['is_complete']:
                print(f"Skipping {task['name']} (already complete)")
                skipped += 1
            else:
                filtered_tasks.append(task)
        tasks = filtered_tasks
        print(f"Skipped {skipped} completed datasets, processing {len(tasks)} videos")
    
    if not tasks:
        print("All videos already preprocessed!")
        return
    
    # Process videos
    results = processor.process_batch(tasks, script_path='scripts/preprocess_data.py')
    
    # Print summary
    processor.print_summary()
    
    # Save results
    processor.save_results(args.results_file)


if __name__ == '__main__':
    main()