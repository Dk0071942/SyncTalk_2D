#!/usr/bin/env python3
"""Test script to verify training state fixes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synctalk.training import TrainingStateManager, SyncNetTrainer

def test_training_state():
    """Test the training state functionality."""
    
    dataset_dir = "./dataset/250627_CB"
    save_dir = "./checkpoint/250627_CB"
    syncnet_save_dir = "./syncnet_ckpt/250627_CB"
    
    print("=== Testing Training State Manager ===\n")
    
    # Test 1: Load and display state
    state_manager = TrainingStateManager(dataset_dir)
    print("1. Current state file contents:")
    print(f"   Preprocessing: {state_manager.state['preprocessing']}")
    print(f"   SyncNet: {state_manager.state['syncnet_training']}")
    print(f"   Main Training: {state_manager.state['main_training']}")
    print()
    
    # Test 2: Check training status
    last_epoch, last_checkpoint = TrainingStateManager.check_training_status(save_dir, dataset_dir)
    print("2. Main training status:")
    print(f"   Last epoch: {last_epoch}")
    print(f"   Last checkpoint: {last_checkpoint}")
    print(f"   Should start from epoch: {last_epoch}")
    print()
    
    # Test 3: Check SyncNet status
    syncnet_trainer = SyncNetTrainer(syncnet_save_dir, dataset_dir, 'ave')
    existing_epochs, existing_checkpoint = syncnet_trainer.check_existing_checkpoint()
    print("3. SyncNet training status:")
    print(f"   Existing epochs: {existing_epochs}")
    print(f"   Existing checkpoint: {existing_checkpoint}")
    print(f"   Is sufficient (>90): {existing_epochs > 90}")
    print()
    
    # Test 4: Display formatted summary (like in train_328.py)
    print("4. Training State Summary (as shown in train_328.py):")
    print(f"   Preprocessing: {'✓ Complete' if state_manager.state['preprocessing']['completed'] else '⚠ Incomplete'}")
    
    syncnet_epochs = state_manager.state['syncnet_training'].get('epochs', 0)
    syncnet_completed = state_manager.state['syncnet_training'].get('completed', False)
    if syncnet_completed:
        print(f"   SyncNet: ✓ Complete ({syncnet_epochs} epochs)")
    else:
        print(f"   SyncNet: ⚠ Incomplete ({syncnet_epochs} epochs)")
    
    print(f"   Main Training: {state_manager.state['main_training']['epochs']} epochs completed")


if __name__ == "__main__":
    test_training_state()