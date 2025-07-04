"""
Training module for SyncTalk 2D.

This module contains all training-related functionality including:
- Loss functions
- Training loops  
- State management
- Checkpoint handling
"""

from .losses import PerceptualLoss, cosine_loss
from .state_manager import TrainingStateManager
from .trainer import ModelTrainer
from .syncnet_trainer import SyncNetTrainer

__all__ = [
    'PerceptualLoss',
    'cosine_loss', 
    'TrainingStateManager',
    'ModelTrainer',
    'SyncNetTrainer'
]