"""
Training state management for SyncTalk 2D.
"""

import os
import json
import time
import glob
from typing import Dict, Any, Tuple, Optional, List


class TrainingStateManager:
    """Manages training state persistence and recovery."""
    
    def __init__(self, dataset_dir: str):
        """
        Initialize state manager.
        
        Args:
            dataset_dir: Base directory for the dataset
        """
        self.dataset_dir = dataset_dir
        self.state_file = os.path.join(dataset_dir, '.training_state.json')
        self.state = self.load_state()
    
    def load_state(self) -> Dict[str, Any]:
        """Load training state from file."""
        # Default state structure
        default_state = {
            "preprocessing": {"completed": False, "timestamp": None},
            "syncnet_training": {"completed": False, "epochs": 0, "checkpoint": None, "timestamp": None},
            "main_training": {"completed": False, "epochs": 0, "checkpoints": [], "timestamp": None}
        }
        
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    
                # Merge loaded state with defaults to ensure all keys exist
                for key, default_value in default_state.items():
                    if key not in loaded_state:
                        loaded_state[key] = default_value
                    elif isinstance(default_value, dict):
                        # Merge nested dictionaries
                        for sub_key, sub_default in default_value.items():
                            if sub_key not in loaded_state[key]:
                                loaded_state[key][sub_key] = sub_default
                
                return loaded_state
            except Exception as e:
                print(f"[WARNING] Failed to load state file: {e}. Using defaults.")
        
        return default_state
    
    def save_state(self):
        """Save current state to file."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update_preprocessing(self, completed: bool = True, **metadata):
        """Update preprocessing status."""
        self.state['preprocessing']['completed'] = completed
        self.state['preprocessing']['timestamp'] = time.time()
        self.state['preprocessing'].update(metadata)
        self.save_state()
    
    def update_syncnet_training(self, epochs: int, checkpoint: Optional[str] = None, completed: bool = False):
        """Update SyncNet training progress."""
        self.state['syncnet_training']['epochs'] = epochs
        if checkpoint:
            self.state['syncnet_training']['checkpoint'] = checkpoint
        self.state['syncnet_training']['completed'] = completed
        self.state['syncnet_training']['timestamp'] = time.time()
        self.save_state()
    
    def update_main_training(self, epochs: int, checkpoint: Optional[str] = None, completed: bool = False):
        """Update main model training progress."""
        self.state['main_training']['epochs'] = epochs
        if checkpoint and checkpoint not in self.state['main_training']['checkpoints']:
            self.state['main_training']['checkpoints'].append(checkpoint)
            # Keep only the last 10 checkpoints in the list to prevent unbounded growth
            if len(self.state['main_training']['checkpoints']) > 10:
                self.state['main_training']['checkpoints'] = self.state['main_training']['checkpoints'][-10:]
        self.state['main_training']['completed'] = completed
        self.state['main_training']['timestamp'] = time.time()
        self.save_state()
    
    def is_syncnet_sufficiently_trained(self, min_epochs: int = 90) -> bool:
        """
        Check if SyncNet training is sufficient (> min_epochs).
        This doesn't mean training should stop, just that existing checkpoint is good enough.
        
        Args:
            min_epochs: Minimum epochs to consider training sufficient (default: 90)
            
        Returns:
            True if training is sufficient to skip retraining
        """
        return (self.state['syncnet_training']['completed'] or 
                self.state['syncnet_training']['epochs'] > min_epochs)
    
    def is_main_training_sufficient(self, min_epochs: int = 90) -> bool:
        """
        Check if main training is sufficient (> min_epochs).
        This doesn't mean training should stop, just that existing checkpoint is good enough.
        
        Args:
            min_epochs: Minimum epochs to consider training sufficient (default: 90)
            
        Returns:
            True if training is sufficient to skip retraining
        """
        return (self.state['main_training']['completed'] or 
                self.state['main_training']['epochs'] > min_epochs)
    
    @staticmethod
    def check_preprocessed_data(dataset_name: str, dataset_dir: str = 'dataset', asr_mode: str = 'ave') -> Tuple[bool, str]:
        """
        Check if preprocessed data exists for the given dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_dir: Base directory for datasets
            asr_mode: ASR model type
            
        Returns:
            (is_valid, message)
        """
        dataset_path = os.path.join(dataset_dir, dataset_name)
        
        required_dirs = ['full_body_img', 'landmarks']
        audio_feature_files = {
            'ave': 'aud_ave.npy',
            'hubert': 'aud_hubert.npy',
            'wenet': 'aud_wenet.npy'
        }
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if not os.path.exists(dir_path) or not os.listdir(dir_path):
                return False, f"Missing or empty directory: {dir_path}"
        
        # Check specific audio feature file
        audio_feature_file = audio_feature_files.get(asr_mode, 'aud_ave.npy')
        audio_feature_path = os.path.join(dataset_path, audio_feature_file)
        if not os.path.exists(audio_feature_path):
            return False, f"Missing audio feature file: {audio_feature_path}"
        
        # Check audio wav
        audio_wav = os.path.join(dataset_path, 'aud.wav')
        if not os.path.exists(audio_wav):
            return False, f"Missing audio file: {audio_wav}"
        
        # Count files for info
        frame_count = len(os.listdir(os.path.join(dataset_path, 'full_body_img')))
        landmark_count = len(os.listdir(os.path.join(dataset_path, 'landmarks')))
        
        # Update state file
        state_manager = TrainingStateManager(dataset_path)
        state_manager.update_preprocessing(
            completed=True,
            frame_count=frame_count,
            landmark_count=landmark_count
        )
        
        return True, f"Preprocessed data found: {frame_count} frames, {landmark_count} landmarks"
    
    @staticmethod
    def check_training_status(save_dir: str, dataset_dir: Optional[str] = None) -> Tuple[int, Optional[str]]:
        """
        Check existing training progress.
        
        Args:
            save_dir: Directory where checkpoints are saved
            dataset_dir: Dataset directory (for state file)
            
        Returns:
            (last_epoch, last_checkpoint_path)
        """
        # First check state file if dataset_dir provided
        if dataset_dir:
            state_manager = TrainingStateManager(dataset_dir)
            main_epochs = state_manager.state.get('main_training', {}).get('epochs', 0)
            if main_epochs > 0:
                # Try to find the most recent existing checkpoint
                # Since we keep only 3 physical files, check backwards from the current epoch
                for check_epoch in range(main_epochs - 1, max(main_epochs - 5, -1), -1):
                    checkpoint_path = os.path.join(save_dir, f"{check_epoch}.pth")
                    if os.path.exists(checkpoint_path):
                        # Return the checkpoint epoch + 1 as the starting epoch
                        return check_epoch + 1, checkpoint_path
        
        # Fallback to checking filesystem
        if not os.path.exists(save_dir):
            return 0, None
        
        checkpoints = glob.glob(os.path.join(save_dir, '*.pth'))
        if not checkpoints:
            return 0, None
        
        # Find the latest checkpoint by epoch number
        latest_epoch = -1
        latest_checkpoint = None
        for ckpt in checkpoints:
            try:
                epoch_num = int(os.path.basename(ckpt).split('.')[0])
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint = ckpt
            except:
                continue
        
        return latest_epoch + 1 if latest_epoch >= 0 else 0, latest_checkpoint
    
    @staticmethod
    def manage_checkpoints(save_dir: str, current_epoch: int, max_checkpoints: int = 3):
        """
        Manage checkpoints to keep only the most recent ones.
        
        Args:
            save_dir: Directory where checkpoints are saved
            current_epoch: Current epoch number
            max_checkpoints: Maximum number of checkpoints to keep
        """
        if not os.path.exists(save_dir):
            return
        
        # Get all checkpoint files
        checkpoints = []
        for ckpt in glob.glob(os.path.join(save_dir, '*.pth')):
            try:
                epoch_num = int(os.path.basename(ckpt).split('.')[0])
                checkpoints.append((epoch_num, ckpt))
            except:
                continue
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: x[0])
        
        # Keep only the most recent checkpoints
        if len(checkpoints) > max_checkpoints:
            # Always keep the latest checkpoint
            checkpoints_to_remove = checkpoints[:-max_checkpoints]
            for epoch, ckpt_path in checkpoints_to_remove:
                # Don't remove the current epoch's checkpoint
                if epoch != current_epoch:
                    try:
                        os.remove(ckpt_path)
                        print(f"[INFO] Removed old checkpoint: {ckpt_path}")
                    except:
                        pass