"""
SyncNet trainer for SyncTalk 2D.
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import glob
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..core.syncnet_328 import SyncNet_color, Dataset
from .losses import cosine_loss
from .state_manager import TrainingStateManager


class SyncNetTrainer:
    """Trainer for SyncNet model."""
    
    def __init__(self, save_dir: str, dataset_dir: str, asr_mode: str = 'ave'):
        """
        Initialize SyncNet trainer.
        
        Args:
            save_dir: Directory to save checkpoints (e.g., ./syncnet_ckpt/name)
            dataset_dir: Dataset directory
            asr_mode: ASR model type
        """
        self.save_dir = save_dir  # Use the provided directory directly
        self.dataset_dir = dataset_dir
        self.asr_mode = asr_mode
        self.state_manager = TrainingStateManager(dataset_dir)
        self.loss_history = []  # Track loss history
        
    def check_existing_checkpoint(self) -> tuple:
        """
        Check for existing SyncNet checkpoint.
        
        Returns:
            (epochs_completed, checkpoint_path)
        """
        # Check state first
        state_epochs = self.state_manager.state['syncnet_training'].get('epochs', 0)
        state_checkpoint = self.state_manager.state['syncnet_training'].get('checkpoint')
        
        # If we have state info and the checkpoint exists, trust it
        if state_epochs > 0 and state_checkpoint and os.path.exists(state_checkpoint):
            return state_epochs, state_checkpoint
        
        # Otherwise, check filesystem for latest checkpoint
        checkpoints = []
        if os.path.exists(self.save_dir):
            checkpoints = glob.glob(os.path.join(self.save_dir, '*.pth'))
        
        if checkpoints:
            # Find the latest checkpoint
            latest_epoch = 0
            latest_checkpoint = None
            
            for ckpt in checkpoints:
                try:
                    # Try to extract epoch from filename
                    epoch_num = int(os.path.basename(ckpt).split('.')[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint = ckpt
                except:
                    continue
            
            if latest_checkpoint:
                # Update state to reflect found checkpoint
                self.state_manager.update_syncnet_training(
                    epochs=latest_epoch,
                    checkpoint=latest_checkpoint,
                    completed=False
                )
                return latest_epoch, latest_checkpoint
        
        return 0, None
    
    def train(self, epochs: int = 100, force: bool = False):
        """
        Train SyncNet model.
        
        Args:
            epochs: Number of epochs to train
            force: Force retraining even if checkpoint exists
            
        Returns:
            Path to best checkpoint
        """
        # Check existing checkpoint
        existing_epochs, existing_checkpoint = self.check_existing_checkpoint()
        if existing_epochs >= 90 and not force:
            print(f"[INFO] SyncNet already trained ({existing_epochs} epochs)")
            print(f"[INFO] Using checkpoint: {existing_checkpoint}")
            return existing_checkpoint
        
        # Determine starting epoch
        start_epoch = 0
        if existing_epochs > 0 and existing_checkpoint and not force:
            start_epoch = existing_epochs
            print(f"[INFO] Continuing SyncNet training from epoch {start_epoch}")
            print(f"[INFO] Loading checkpoint: {existing_checkpoint}")
        else:
            print(f'[INFO] Starting SyncNet training from scratch...')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup dataset and dataloader
        train_dataset = Dataset(self.dataset_dir, mode=self.asr_mode)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True,
            num_workers=16
        )
        
        # Setup model and optimizer
        model = SyncNet_color(self.asr_mode).cuda()
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001
        )
        
        # Load existing checkpoint if continuing
        best_loss = float('inf')
        best_checkpoint = existing_checkpoint if start_epoch > 0 else None
        
        if start_epoch > 0 and existing_checkpoint and os.path.exists(existing_checkpoint):
            try:
                # Load model state
                checkpoint_data = torch.load(existing_checkpoint)
                if isinstance(checkpoint_data, dict):
                    # Handle checkpoint with metadata
                    model.load_state_dict(checkpoint_data.get('model_state_dict', checkpoint_data))
                    if 'optimizer_state_dict' in checkpoint_data:
                        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    if 'best_loss' in checkpoint_data:
                        best_loss = checkpoint_data['best_loss']
                else:
                    # Handle simple state dict
                    model.load_state_dict(checkpoint_data)
                
                print(f"[INFO] Loaded checkpoint from epoch {existing_epochs}")
                
                # Try to load loss history if available
                loss_history_file = os.path.join(self.save_dir, 'syncnet_training_log.json')
                if os.path.exists(loss_history_file):
                    with open(loss_history_file, 'r') as f:
                        self.loss_history = json.load(f)
                        # Keep only history up to start_epoch
                        self.loss_history = [h for h in self.loss_history if h['epoch'] <= start_epoch]
                        print(f"[INFO] Loaded training history with {len(self.loss_history)} epochs")
                
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                print("[INFO] Starting from scratch instead")
                start_epoch = 0
                best_checkpoint = None
        
        # Training loop - continue from start_epoch to epochs
        pbar = tqdm(range(start_epoch, epochs), desc='SyncNet Training')
        for epoch in pbar:
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                imgT, audioT, y = batch
                imgT = imgT.cuda()
                audioT = audioT.cuda()
                y = y.cuda()
                
                # Forward pass
                audio_embedding, face_embedding = model(imgT, audioT)
                loss = cosine_loss(audio_embedding, face_embedding, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            pbar.set_description(f'SyncNet training epoch: {epoch+1}, loss: {avg_loss:.4f}')
            
            # Track loss history
            self.loss_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'timestamp': time.time()
            })
            
            # Save checkpoint ONLY if loss improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # Remove old checkpoint if exists
                if best_checkpoint and os.path.exists(best_checkpoint):
                    try:
                        os.remove(best_checkpoint)
                    except:
                        pass
                
                # Ensure directory exists
                os.makedirs(self.save_dir, exist_ok=True)
                
                checkpoint_path = os.path.join(self.save_dir, f'{epoch+1}.pth')
                # Save checkpoint with metadata for better continuation
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'loss_history': self.loss_history[-10:]  # Keep last 10 epochs for reference
                }, checkpoint_path)
                best_checkpoint = checkpoint_path
                print(f'  [INFO] Better loss! Saved checkpoint: {checkpoint_path}')
                
                # Update state with best loss info
                self.state_manager.state['syncnet_training']['best_loss'] = best_loss
                self.state_manager.update_syncnet_training(
                    epochs=epoch + 1,
                    checkpoint=checkpoint_path,
                    completed=(epoch + 1 >= epochs)
                )
                
                # Manage checkpoints (keep only 3 most recent)
                TrainingStateManager.manage_checkpoints(self.save_dir, epoch, max_checkpoints=3)
        
        print(f'SyncNet training finished...')
        
        # Save loss history
        self.save_loss_history()
        
        # Mark as completed
        self.state_manager.update_syncnet_training(
            epochs=epochs,
            checkpoint=best_checkpoint,
            completed=True
        )
        
        return best_checkpoint
    
    def save_loss_history(self):
        """
        Save SyncNet loss history as both JSON log and visual graph.
        """
        if not self.loss_history:
            return
        
        # Save as JSON log
        log_path = os.path.join(self.save_dir, 'syncnet_training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"[INFO] Saved SyncNet training log to {log_path}")
        
        # Save as graph
        plt.figure(figsize=(10, 6))
        epochs = [item['epoch'] for item in self.loss_history]
        losses = [item['loss'] for item in self.loss_history]
        
        plt.plot(epochs, losses, 'g-', linewidth=2, label='SyncNet Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Cosine Loss', fontsize=12)
        plt.title('SyncNet Training Loss History', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add some statistics
        min_loss = min(losses)
        min_epoch = epochs[losses.index(min_loss)]
        plt.axhline(y=min_loss, color='r', linestyle='--', alpha=0.5)
        plt.text(epochs[-1] * 0.02, min_loss * 1.01, f'Min: {min_loss:.6f} @ epoch {min_epoch}', 
                 fontsize=10, color='red')
        
        graph_path = os.path.join(self.save_dir, 'syncnet_loss_history.png')
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Saved SyncNet loss graph to {graph_path}")