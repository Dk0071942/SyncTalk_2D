#!/usr/bin/env python3
"""
Training script for SyncTalk_2D 328x328 models.

This is a CLI wrapper that uses the core training functionality from synctalk.training.
"""

import argparse
import os
import sys
import torch
import subprocess
import glob

# Add parent directory to path so we can import synctalk module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synctalk.core.unet_328 import Model
from synctalk.training import ModelTrainer, TrainingStateManager, SyncNetTrainer


def get_args():
    parser = argparse.ArgumentParser(description='Train SyncTalk_2D 328x328 model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Dataset configuration
    parser.add_argument('--name', type=str, help='Dataset name (for train_only mode)')
    parser.add_argument('--dataset_dir', type=str, help='Full dataset directory path (for compatibility)')
    parser.add_argument('--dataset_base', type=str, default='dataset', help='Base directory for datasets')
    
    # Model configuration
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="", help="Path to syncnet checkpoint")
    parser.add_argument('--train_syncnet', action='store_true', help='Train SyncNet before main model')
    parser.add_argument('--syncnet_checkpoint_path', type=str, help='Pre-trained SyncNet checkpoint path')
    
    # Training configuration
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Base checkpoint directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, help='Alias for epochs')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--batch_size', type=int, help='Alias for batchsize')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="ave")
    
    # Other options
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--continue_training', action='store_true', help='Continue from latest checkpoint')
    parser.add_argument('--gpu', type=str, help='GPU device ID (already set via CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--force', action='store_true', help='Force training even if already sufficient')

    return parser.parse_args()


def main():
    args = get_args()
    
    # Handle argument aliases and determine dataset path
    if args.batch_size is not None:
        args.batchsize = args.batch_size
    if args.num_epochs is not None:
        args.epochs = args.num_epochs
    
    # Determine dataset directory
    if args.name and not args.dataset_dir:
        args.dataset_dir = os.path.join(args.dataset_base, args.name)
    
    # Check if preprocessed data exists
    data_exists, message = TrainingStateManager.check_preprocessed_data(
        args.name if args.name else os.path.basename(args.dataset_dir),
        args.dataset_base,
        args.asr
    )
    if not data_exists:
        print(f"Error: {message}")
        print(f"\nPlease run preprocessing first:")
        print(f"  python scripts/preprocess_data.py --video_path YOUR_VIDEO.mp4 --name {args.name} --asr_model {args.asr}")
        sys.exit(1)
    else:
        print(f"[INFO] {message}")
    
    # Determine save directory
    if not args.save_dir:
        if args.name:
            args.save_dir = os.path.join(args.checkpoint_dir, args.name)
        else:
            print("Error: Either --save_dir or --name must be specified")
            sys.exit(1)
    
    # Load training state
    state_manager = TrainingStateManager(args.dataset_dir)
    
    # Handle SyncNet training
    if args.train_syncnet and not args.syncnet_checkpoint_path:
        # Use the traditional syncnet_ckpt/name directory structure
        syncnet_save_dir = os.path.join('./syncnet_ckpt', args.name)
        syncnet_trainer = SyncNetTrainer(syncnet_save_dir, args.dataset_dir, args.asr)
        
        # Check if SyncNet training is needed
        existing_epochs, existing_checkpoint = syncnet_trainer.check_existing_checkpoint()
        if existing_epochs > 90 and not args.force:
            args.syncnet_checkpoint = existing_checkpoint
            args.use_syncnet = True
            print(f"=== Using existing SyncNet (epoch {existing_epochs} > 90) ===")
            print(f"Checkpoint: {args.syncnet_checkpoint}")
            print(f"[INFO] Skipping SyncNet training as it's already sufficient.")
        else:
            # Train SyncNet for full 100 epochs
            print("=== Training SyncNet first ===")
            if existing_epochs > 0:
                print(f"[INFO] Found existing SyncNet at epoch {existing_epochs}, continuing to 100 epochs...")
            args.syncnet_checkpoint = syncnet_trainer.train(epochs=100, force=args.force)
            args.use_syncnet = True
    elif args.syncnet_checkpoint_path:
        args.syncnet_checkpoint = args.syncnet_checkpoint_path
        args.use_syncnet = True
    
    # Print configuration
    print(f"\n=== Training Configuration ===")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batchsize}")
    print(f"Learning rate: {args.lr}")
    print(f"ASR model: {args.asr}")
    print(f"Use SyncNet: {args.use_syncnet}")
    if args.use_syncnet:
        print(f"SyncNet checkpoint: {args.syncnet_checkpoint}")
    
    # Check existing training status
    last_epoch, last_checkpoint = TrainingStateManager.check_training_status(args.save_dir, args.dataset_dir)
    if last_epoch > 0:
        print(f"\n[INFO] Found existing training progress: {last_epoch-1} epochs completed")
        print(f"[INFO] Last checkpoint: {last_checkpoint}")
    
    # Show training state summary
    print(f"\n=== Training State Summary ===")
    print(f"Preprocessing: {'✓ Complete' if state_manager.state['preprocessing']['completed'] else '⚠ Incomplete'}")
    
    # Fix SyncNet display to show actual epochs from state
    syncnet_epochs = state_manager.state['syncnet_training'].get('epochs', 0)
    syncnet_completed = state_manager.state['syncnet_training'].get('completed', False)
    if syncnet_completed:
        print(f"SyncNet: ✓ Complete ({syncnet_epochs} epochs)")
    else:
        print(f"SyncNet: ⚠ Incomplete ({syncnet_epochs} epochs)")
    
    print(f"Main Training: {state_manager.state['main_training']['epochs']} epochs completed")
    print()
    
    # Initialize model
    net = Model(6, mode=args.asr).cuda()
    
    # Handle continue training - automatically continue if checkpoint exists
    start_epoch = 0
    if last_checkpoint and os.path.exists(last_checkpoint):
        # Automatically continue from last checkpoint if it exists
        print(f"\n[INFO] Continuing from checkpoint: {last_checkpoint} (epoch {last_epoch})")
        net.load_state_dict(torch.load(last_checkpoint, weights_only=True))
        start_epoch = last_epoch
        
        # Check if we've already reached the target epochs
        if start_epoch >= args.epochs:
            print(f"\n[INFO] Training already completed ({start_epoch} epochs >= target {args.epochs} epochs)")
            if not args.force:
                print("[INFO] Use --force to continue training beyond target epochs.")
                sys.exit(0)
    elif args.continue_training and not last_checkpoint:
        print(f"\n[WARNING] --continue_training specified but no checkpoint found. Starting from scratch.")
    
    # Train using ModelTrainer
    trainer = ModelTrainer(args)
    trainer.train(net, start_epoch=start_epoch)
    
    print("\n=== Training completed! ===")


if __name__ == '__main__':
    main()