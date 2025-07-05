"""
Main trainer class for SyncTalk 2D models.
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import cv2
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from ..core.datasetsss_328 import MyDataset
from ..core.syncnet_328 import SyncNet_color
from .losses import PerceptualLoss, cosine_loss
from .state_manager import TrainingStateManager


class ModelTrainer:
    """Main trainer for SyncTalk 2D models."""
    
    def __init__(self, args):
        """
        Initialize trainer.
        
        Args:
            args: Training arguments namespace
        """
        self.args = args
        self.state_manager = TrainingStateManager(args.dataset_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_history = []  # Track loss history for visualization
        
    def setup_training(self, net):
        """
        Setup training components.
        
        Args:
            net: The model to train
            
        Returns:
            Tuple of (content_loss, syncnet, optimizer, criterion, dataloaders)
        """
        # Loss functions
        content_loss = PerceptualLoss(torch.nn.MSELoss())
        
        # Setup SyncNet if needed
        syncnet = None
        if self.args.use_syncnet:
            if not self.args.syncnet_checkpoint:
                raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'")
            
            syncnet = SyncNet_color(self.args.asr).eval().cuda()
            # Load checkpoint, handling both old and new formats
            checkpoint_data = torch.load(self.args.syncnet_checkpoint, weights_only=True)
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                # New format with metadata
                syncnet.load_state_dict(checkpoint_data['model_state_dict'])
            else:
                # Old format or direct state dict
                syncnet.load_state_dict(checkpoint_data)
        
        # Create save directory
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        # Setup datasets and dataloaders
        dataloader_list = []
        dataset_list = []
        dataset_dir_list = [self.args.dataset_dir]
        
        for dataset_dir in dataset_dir_list:
            dataset = MyDataset(dataset_dir, self.args.asr)
            train_dataloader = DataLoader(
                dataset, 
                batch_size=self.args.batchsize, 
                shuffle=True, 
                drop_last=False, 
                num_workers=16
            )
            dataloader_list.append(train_dataloader)
            dataset_list.append(dataset)
        
        # Optimizer and criterion
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        criterion = nn.L1Loss()
        
        return content_loss, syncnet, optimizer, criterion, (dataloader_list, dataset_list)
    
    def train_epoch(self, net, epoch, content_loss, syncnet, optimizer, criterion, dataloaders):
        """
        Train for one epoch.
        
        Args:
            net: Model to train
            epoch: Current epoch number
            content_loss: Perceptual loss function
            syncnet: SyncNet model (optional)
            optimizer: Optimizer
            criterion: L1 loss criterion
            dataloaders: Tuple of (dataloader_list, dataset_list)
        """
        dataloader_list, dataset_list = dataloaders
        net.train()
        
        # Select random dataset if multiple
        random_i = random.randint(0, len(dataset_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]
        
        epoch_losses = []  # Track losses for this epoch
        
        with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img', leave=False, position=0, ncols=100) as pbar:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                
                # Forward pass
                preds = net(imgs, audio_feat)
                
                # Calculate losses
                loss_pixel = criterion(preds, labels)
                loss_perceptual = content_loss.get_loss(preds, labels)
                
                if syncnet is not None:
                    y = torch.ones([preds.shape[0], 1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                    loss = loss_pixel + loss_perceptual * 0.01 + 10 * sync_loss
                else:
                    loss = loss_pixel + loss_perceptual * 0.01
                
                epoch_losses.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
        
        # Store average epoch loss
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.loss_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'timestamp': time.time()
            })
            print(f"  [INFO] Epoch {epoch + 1} average loss: {avg_loss:.6f}")
    
    def save_checkpoint(self, net, epoch):
        """
        Save model checkpoint.
        
        Args:
            net: Model to save
            epoch: Current epoch number
        """
        checkpoint_path = os.path.join(self.args.save_dir, f"{epoch}.pth")
        torch.save(net.state_dict(), checkpoint_path)
        print(f"  [INFO] Checkpoint saved: {epoch}.pth")
        
        # Update training state
        self.state_manager.update_main_training(
            epochs=epoch + 1,
            checkpoint=checkpoint_path,
            completed=(epoch + 1 >= self.args.epochs)
        )
        
        # Manage checkpoints (keep only 3 most recent)
        # This keeps the 3 highest epoch checkpoints, not necessarily the best performing
        TrainingStateManager.manage_checkpoints(self.args.save_dir, epoch, max_checkpoints=3)
    
    def save_visualization(self, net, dataset, epoch):
        """
        Save visualization results during training.
        
        Args:
            net: Model
            dataset: Dataset for sampling
            epoch: Current epoch number
        """
        if not self.args.see_res:
            return
            
        net.eval()
        
        # Sample random data
        idx = random.randint(0, len(dataset) - 1)
        img_concat_T, img_real_T, audio_feat = dataset[idx]
        img_concat_T = img_concat_T[None].cuda()
        audio_feat = audio_feat[None].cuda()
        
        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)[0]
        
        # Convert to numpy and save
        pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)
        img_real = img_real_T.numpy().transpose(1, 2, 0) * 255
        img_real = np.array(img_real, dtype=np.uint8)
        
        os.makedirs("./train_tmp_img", exist_ok=True)
        cv2.imwrite(f"./train_tmp_img/epoch_{epoch}.jpg", pred)
        cv2.imwrite(f"./train_tmp_img/epoch_{epoch}_real.jpg", img_real)
    
    def train(self, net, start_epoch=0):
        """
        Main training loop.
        
        Args:
            net: Model to train
            start_epoch: Starting epoch (for resuming)
        """
        # Check if training is already sufficient (but still continue to target epochs)
        if self.state_manager.is_main_training_sufficient(min_epochs=90):
            print(f"[INFO] Training already sufficient ({self.state_manager.state['main_training']['epochs']} epochs completed)")
            if not self.args.force:
                print("[INFO] Skipping training. Use --force to retrain.")
                return
        
        # Setup training components
        content_loss, syncnet, optimizer, criterion, dataloaders = self.setup_training(net)
        dataset_list = dataloaders[1]
        
        # Training loop - always go to full epochs
        for epoch in range(start_epoch, self.args.epochs):
            # Train one epoch
            self.train_epoch(net, epoch, content_loss, syncnet, optimizer, criterion, dataloaders)
            
            # Save checkpoint after every epoch
            self.save_checkpoint(net, epoch)
            
            # Save visualization
            self.save_visualization(net, dataset_list[0], epoch)
        
        # Save loss history after training completes
        self.save_loss_history()
    
    def save_loss_history(self):
        """
        Save loss history as both JSON log and visual graph.
        """
        try:
            if not self.loss_history:
                print("[WARNING] No loss history to save")
                return
            
            # Save as JSON log
            log_path = os.path.join(self.args.save_dir, 'training_log.json')
            try:
                with open(log_path, 'w') as f:
                    json.dump(self.loss_history, f, indent=2)
                print(f"[INFO] Saved training log to {log_path}")
            except Exception as e:
                print(f"[WARNING] Failed to save training log: {e}")
            
            # Save as graph - with error handling
            try:
                # Filter out any invalid entries
                valid_entries = [item for item in self.loss_history 
                               if isinstance(item.get('epoch'), (int, float)) 
                               and isinstance(item.get('loss'), (int, float))
                               and not (item.get('loss') is None or item.get('loss') != item.get('loss'))]  # Check for NaN
                
                if not valid_entries:
                    print("[WARNING] No valid loss data to plot")
                    return
                
                plt.figure(figsize=(10, 6))
                epochs = [item['epoch'] for item in valid_entries]
                losses = [item['loss'] for item in valid_entries]
                
                # Ensure we have data to plot
                if len(epochs) == 0 or len(losses) == 0:
                    print("[WARNING] Empty data for plotting")
                    plt.close()
                    return
                
                plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title(f'Training Loss History - {getattr(self.args, "name", "Model")}', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add statistics only if we have valid data
                if losses:
                    min_loss = min(losses)
                    min_epoch = epochs[losses.index(min_loss)]
                    plt.axhline(y=min_loss, color='r', linestyle='--', alpha=0.5)
                    
                    # Safely position text
                    text_x = max(1, epochs[-1] * 0.02) if epochs else 1
                    text_y = min_loss * 1.01 if min_loss > 0 else min_loss + 0.0001
                    plt.text(text_x, text_y, f'Min: {min_loss:.6f} @ epoch {min_epoch}', 
                             fontsize=10, color='red')
                
                graph_path = os.path.join(self.args.save_dir, 'loss_history.png')
                plt.tight_layout()
                plt.savefig(graph_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"[INFO] Saved loss graph to {graph_path}")
                
            except Exception as e:
                print(f"[WARNING] Failed to save loss graph: {e}")
                plt.close()  # Ensure we close the figure even on error
                
        except Exception as e:
            print(f"[ERROR] Unexpected error in save_loss_history: {e}")