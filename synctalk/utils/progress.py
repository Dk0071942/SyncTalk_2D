"""
Progress bar utilities for SyncTalk 2D.

Provides standardized progress bar handling with tqdm to ensure
clean console output without new lines.
"""

from typing import Optional, Callable
from tqdm import tqdm
import sys
import threading


# Global lock for synchronized console output
_console_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print that works well with tqdm progress bars."""
    with _console_lock:
        # Clear the current line first
        tqdm.write("\r" + " " * 100 + "\r", end="")
        # Use tqdm.write for proper coordination with progress bars
        tqdm.write(" ".join(str(arg) for arg in args), **kwargs)


class ProgressBar:
    """A wrapper around tqdm for consistent progress bar behavior."""
    
    def __init__(self, total: int, desc: str = "", unit: str = "it", 
                 leave: bool = False, position: int = 0):
        """
        Initialize progress bar with standardized settings.
        
        Args:
            total: Total number of iterations
            desc: Description to show
            unit: Unit name for iterations
            leave: Whether to leave the progress bar after completion
            position: Position for nested bars (0 = first/only bar)
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            position=position,
            ncols=100,  # Fixed width for consistency
            file=sys.stdout,  # Explicit stdout
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.is_closed = False
    
    def update(self, n: int = 1, msg: Optional[str] = None):
        """
        Update the progress bar.
        
        Args:
            n: Number of steps to increment
            msg: Optional message to display in postfix
        """
        if not self.is_closed:
            if msg:
                self.pbar.set_postfix_str(msg)
            self.pbar.update(n)
    
    def close(self):
        """Close the progress bar."""
        if not self.is_closed:
            self.pbar.close()
            self.is_closed = True
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_progress_callback(desc: str = "Processing", 
                           unit: str = "steps",
                           leave: bool = False) -> tuple:
    """
    Create a progress callback function for use with processors.
    
    Args:
        desc: Description for the progress bar
        unit: Unit name for progress
        leave: Whether to leave the bar after completion
        
    Returns:
        Tuple of (callback_function, progress_bar)
    """
    progress_bar = None
    
    def callback(current: int, total: int, message: str = ""):
        nonlocal progress_bar
        
        # Initialize progress bar on first call
        if progress_bar is None:
            progress_bar = ProgressBar(
                total=total,
                desc=desc,
                unit=unit,
                leave=leave
            )
        
        # Update progress
        if current > 0:
            progress_bar.update(1, msg=message)
        
        # Close when complete
        if current >= total:
            progress_bar.close()
    
    return callback, progress_bar


def wrap_iterable(iterable, desc: str = "Processing", 
                  leave: bool = False, position: int = 0):
    """
    Wrap an iterable with a tqdm progress bar using standardized settings.
    
    Args:
        iterable: The iterable to wrap
        desc: Description for the progress bar
        leave: Whether to leave the bar after completion
        position: Position for nested bars
        
    Returns:
        Wrapped iterable with progress bar
    """
    return tqdm(
        iterable,
        desc=desc,
        leave=leave,
        position=position,
        ncols=100,
        file=sys.stdout,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )