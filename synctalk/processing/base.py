"""
Base classes for video processing in SyncTalk 2D.

This module defines abstract base classes and common interfaces
for all video processors in the system.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import torch


class BaseVideoProcessor(ABC):
    """
    Abstract base class for all video processors in SyncTalk 2D.
    
    This class defines the common interface that all video processors
    must implement, ensuring consistency across different processing modes.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the base video processor.
        
        Args:
            model_name: Name of the model (e.g., "AD2.2", "LS1")
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Common paths
        self.checkpoint_path = Path(f"./checkpoint/{model_name}")
        self.dataset_dir = Path(f"./dataset/{model_name}")
        
        # Common configuration
        self.fps: int = 25  # Default FPS
        self.mode: Optional[str] = None  # Audio feature mode
        
    @abstractmethod
    def load_models(self, checkpoint_number: Optional[int] = None,
                    mode: str = "ave") -> str:
        """
        Load the required models for video generation.
        
        Args:
            checkpoint_number: Specific checkpoint to load. If None, loads latest.
            mode: Audio feature mode ("ave", "hubert", or "wenet")
            
        Returns:
            Path to loaded checkpoint file
        """
        pass
    
    @abstractmethod
    def prepare_audio(self, audio_path: str) -> Any:
        """
        Prepare audio for processing.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Processed audio data (format depends on processor type)
        """
        pass
    
    @abstractmethod
    def generate_video(self, audio_path: str, output_path: str,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None,
                      **kwargs) -> str:
        """
        Generate a complete video from audio.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output video
            progress_callback: Optional progress callback(current, total, message)
            **kwargs: Additional processor-specific parameters
            
        Returns:
            Path to the generated video file
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup any temporary resources.
        
        Subclasses should override this if they create temporary files.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the processor configuration.
        
        Returns:
            Dictionary with processor information
        """
        return {
            "processor_type": self.__class__.__name__,
            "model_name": self.model_name,
            "device": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
            "dataset_dir": str(self.dataset_dir),
            "fps": self.fps,
            "mode": self.mode
        }
    
    def validate_paths(self) -> bool:
        """
        Validate that required paths exist.
        
        Returns:
            True if all required paths exist
        """
        if not self.checkpoint_path.exists():
            print(f"Warning: Checkpoint path not found: {self.checkpoint_path}")
            return False
            
        if not self.dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {self.dataset_dir}")
            return False
            
        return True
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()


class ProgressTracker:
    """
    Helper class for tracking and reporting progress.
    
    This class provides a consistent interface for progress tracking
    across different processors and UI frameworks.
    """
    
    def __init__(self, total_steps: int = 100,
                 callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            callback: Optional progress callback(current, total, message)
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = callback
        self._phase_weights = {}
        self._current_phase = None
        
    def add_phase(self, name: str, weight: float) -> None:
        """
        Add a phase with its relative weight.
        
        Args:
            name: Phase name
            weight: Relative weight (0-1)
        """
        self._phase_weights[name] = weight
        
    def start_phase(self, name: str) -> None:
        """
        Start a new phase.
        
        Args:
            name: Phase name
        """
        self._current_phase = name
        
    def update(self, progress: float, message: str) -> None:
        """
        Update progress within current phase.
        
        Args:
            progress: Progress within phase (0-1)
            message: Progress message
        """
        if self.callback is None:
            return
            
        # Calculate global progress
        global_progress = 0.0
        phase_found = False
        
        for phase_name, weight in self._phase_weights.items():
            if phase_name == self._current_phase:
                global_progress += weight * progress
                phase_found = True
                break
            else:
                global_progress += weight
                
        if not phase_found:
            # Phase not in weights, just use raw progress
            global_progress = progress
            
        # Convert to steps
        current_step = int(global_progress * self.total_steps)
        self.current_step = min(current_step, self.total_steps)
        
        # Call callback
        self.callback(self.current_step, self.total_steps, message)
        
    def complete(self, message: str = "Done!") -> None:
        """
        Mark progress as complete.
        
        Args:
            message: Completion message
        """
        if self.callback:
            self.callback(self.total_steps, self.total_steps, message)