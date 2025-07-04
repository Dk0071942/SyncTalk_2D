"""
Batch processing utilities for SyncTalk 2D.

This module provides utilities for processing multiple videos in batch,
including configuration parsing and result tracking.
"""

import os
import json
import subprocess
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


class BatchProcessor:
    """Handles batch processing of multiple videos."""
    
    def __init__(self, max_workers: int = 1, log_file: Optional[str] = None):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            log_file: Optional log file path
        """
        self.max_workers = max_workers
        self.results = []
        
        # Setup logging
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def load_batch_config(self, config_path: str) -> List[Dict[str, any]]:
        """
        Load batch processing configuration.
        
        Args:
            config_path: Path to configuration file (JSON or text list)
            
        Returns:
            List of processing tasks
        """
        tasks = []
        
        if config_path.endswith('.json'):
            # JSON configuration with detailed options
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Handle both single task and task list
            if isinstance(config, dict):
                tasks = [config]
            elif isinstance(config, list):
                tasks = config
            else:
                raise ValueError("Invalid JSON configuration format")
        else:
            # Text file with video paths (one per line)
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract name from filename
                        video_name = Path(line).stem
                        tasks.append({
                            'video_path': line,
                            'name': video_name
                        })
        
        return tasks
    
    def process_video(self, task: Dict[str, any], script_path: str = 'scripts/preprocess_data.py') -> Dict[str, any]:
        """
        Process a single video.
        
        Args:
            task: Task dictionary with video_path, name, and optional parameters
            script_path: Path to the preprocessing script
            
        Returns:
            Result dictionary with status and timing information
        """
        start_time = time.time()
        video_path = task['video_path']
        name = task['name']
        
        # Build command
        cmd = ['python', script_path, '--video_path', video_path, '--name', name]
        
        # Add optional parameters
        if 'asr_model' in task:
            cmd.extend(['--asr_model', task['asr_model']])
        if 'gpu' in task:
            cmd.extend(['--gpu', str(task['gpu'])])
        if 'extract_audio_only' in task and task['extract_audio_only']:
            cmd.append('--extract_audio_only')
        
        # Run preprocessing
        try:
            self.logger.info(f"Processing: {name} ({video_path})")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed: {name} in {elapsed_time:.1f}s")
            
            return {
                'name': name,
                'video_path': video_path,
                'status': 'success',
                'elapsed_time': elapsed_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Failed: {name} - {e}")
            
            return {
                'name': name,
                'video_path': video_path,
                'status': 'failed',
                'elapsed_time': elapsed_time,
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error: {name} - {e}")
            
            return {
                'name': name,
                'video_path': video_path,
                'status': 'error',
                'elapsed_time': elapsed_time,
                'error': str(e)
            }
    
    def process_batch(self, tasks: List[Dict[str, any]], script_path: str = 'scripts/preprocess_data.py') -> List[Dict[str, any]]:
        """
        Process multiple videos in batch.
        
        Args:
            tasks: List of task dictionaries
            script_path: Path to the preprocessing script
            
        Returns:
            List of result dictionaries
        """
        self.results = []
        total_tasks = len(tasks)
        
        self.logger.info(f"Starting batch processing of {total_tasks} videos with {self.max_workers} workers")
        
        if self.max_workers == 1:
            # Sequential processing
            for i, task in enumerate(tasks):
                self.logger.info(f"[{i+1}/{total_tasks}] Processing {task['name']}")
                result = self.process_video(task, script_path)
                self.results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.process_video, task, script_path): task 
                    for task in tasks
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_task):
                    completed += 1
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        self.results.append(result)
                        self.logger.info(f"[{completed}/{total_tasks}] Completed: {task['name']}")
                    except Exception as e:
                        self.logger.error(f"[{completed}/{total_tasks}] Exception for {task['name']}: {e}")
                        self.results.append({
                            'name': task['name'],
                            'video_path': task['video_path'],
                            'status': 'exception',
                            'error': str(e)
                        })
        
        return self.results
    
    def save_results(self, output_path: str) -> None:
        """
        Save processing results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        # Summary statistics
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = sum(1 for r in self.results if r['status'] in ['failed', 'error', 'exception'])
        total_time = sum(r.get('elapsed_time', 0) for r in self.results)
        
        summary = {
            'total_videos': len(self.results),
            'successful': successful,
            'failed': failed,
            'total_time_seconds': total_time,
            'average_time_seconds': total_time / len(self.results) if self.results else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
        self.logger.info(f"Summary: {successful} successful, {failed} failed out of {len(self.results)} total")
    
    def print_summary(self) -> None:
        """Print a summary of the batch processing results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        successful = []
        failed = []
        
        for result in self.results:
            if result['status'] == 'success':
                successful.append(result)
            else:
                failed.append(result)
        
        print(f"\nTotal videos: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n✓ Successful:")
            for r in successful:
                print(f"  - {r['name']} ({r['elapsed_time']:.1f}s)")
        
        if failed:
            print("\n✗ Failed:")
            for r in failed:
                error = r.get('error', 'Unknown error')
                print(f"  - {r['name']}: {error}")
        
        total_time = sum(r.get('elapsed_time', 0) for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0
        
        print(f"\nTotal processing time: {total_time:.1f}s")
        print(f"Average time per video: {avg_time:.1f}s")
        print("="*60)


def create_batch_config_template(output_path: str) -> None:
    """
    Create a template batch configuration file.
    
    Args:
        output_path: Path to save the template
    """
    template = [
        {
            "video_path": "/path/to/video1.mp4",
            "name": "speaker1",
            "asr_model": "ave",
            "gpu": 0
        },
        {
            "video_path": "/path/to/video2.mp4", 
            "name": "speaker2",
            "asr_model": "hubert",
            "gpu": 0
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Batch configuration template saved to: {output_path}")
    print("Edit this file with your video paths and run batch preprocessing")