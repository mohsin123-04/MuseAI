"""
Training Script for MuseAI Style Transfer Model
Full training loop with checkpointing, logging, and evaluation.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.config import (
    create_directories, get_device, DEVICE, NUM_GPUS,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WARMUP_EPOCHS,
    CONTENT_WEIGHT, STYLE_WEIGHT, IDENTITY_WEIGHT, TV_WEIGHT,
    LR_DECAY_FACTOR, LR_DECAY_EPOCHS,
    CHECKPOINTS_DIR, LOGS_DIR, OUTPUTS_DIR,
    ADAM_BETAS, WEIGHT_DECAY,
    LOG_FREQUENCY, SAMPLE_FREQUENCY, SAVE_FREQUENCY, KEEP_LAST_N_CHECKPOINTS,
    USE_AMP
)
from src.models.style_transfer import StyleTransferNetwork, StyleTransferLoss
from src.models.identity import IdentityLoss
from src.utils.data_loader import create_dataloaders, RawStyleDataset, StyleDataset
from src.utils.metrics import compute_metrics


class Trainer:
    """
    Trainer class for MuseAI style transfer model.
    Handles training loop, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        model: StyleTransferNetwork,
        device: torch.device = DEVICE,
        use_amp: bool = USE_AMP,
        use_conditional: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.use_conditional = use_conditional
        
        # Loss functions
        self.criterion = StyleTransferLoss(
            content_weight=CONTENT_WEIGHT,
            style_weight=STYLE_WEIGHT,
            tv_weight=TV_WEIGHT
        ).to(device)
        
        self.identity_loss_fn = IdentityLoss(weight=IDENTITY_WEIGHT).to(device)
        
        # Optimizer (only for decoder + conditional adain)
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optim.Adam(
            trainable_params,
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            weight_decay=WEIGHT_DECAY
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if use_amp else None
        
        # Checkpointing
        self.checkpoints_dir = Path(CHECKPOINTS_DIR)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logs_dir = Path(LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Outputs
        self.outputs_dir = Path(OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.total_steps = 0
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'content_loss': [],
            'style_loss': [],
            'identity_loss': [],
            'learning_rate': []
        }
        
        # Multi-GPU support
        if NUM_GPUS > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using {NUM_GPUS} GPUs")
    
    def get_model(self):
        """Get the model (handles DataParallel wrapping)."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model
    
    def update_learning_rate(self, epoch: int):
        """Update learning rate based on schedule."""
        lr = LEARNING_RATE
        
        for milestone, factor in zip(LR_DECAY_EPOCHS, [LR_DECAY_FACTOR] * len(LR_DECAY_EPOCHS)):
            if epoch >= milestone:
                lr *= factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        loss_dict = {
            'content_loss': 0.0,
            'style_loss': 0.0,
            'identity_loss': 0.0,
            'tv_loss': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            content = batch['content'].to(self.device)
            style = batch['style'].to(self.device)
            artist_idx = batch['artist_idx'].to(self.device)
            
            # Forward pass
            if self.use_amp and self.scaler is not None:
                with autocast():
                    stylized, features = self.model(
                        content, style, artist_idx,
                        return_features=True
                    )
                    
                    # Main loss
                    loss, loss_dict_batch = self.criterion(
                        stylized, content, style, features['adain_features']
                    )
                    
                    # Identity loss
                    id_loss, id_metrics = self.identity_loss_fn(content, stylized)
                    total_batch_loss = loss + id_loss
            else:
                stylized, features = self.model(
                    content, style, artist_idx,
                    return_features=True
                )
                
                loss, loss_dict_batch = self.criterion(
                    stylized, content, style, features['adain_features']
                )
                
                id_loss, id_metrics = self.identity_loss_fn(content, stylized)
                total_batch_loss = loss + id_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            for key in loss_dict.keys():
                if key in loss_dict_batch:
                    loss_dict[key] += loss_dict_batch[key]
            
            self.total_steps += 1
            
            # Logging
            if (batch_idx + 1) % LOG_FREQUENCY == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Average losses
        num_batches = len(train_loader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'content_loss': loss_dict['content_loss'] / num_batches,
            'style_loss': loss_dict['style_loss'] / num_batches,
            'identity_loss': loss_dict['identity_loss'] / num_batches,
            'tv_loss': loss_dict['tv_loss'] / num_batches,
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation dataloader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                content = batch['content'].to(self.device)
                style = batch['style'].to(self.device)
                artist_idx = batch['artist_idx'].to(self.device)
                
                # Forward pass
                stylized, features = self.model(
                    content, style, artist_idx,
                    return_features=True
                )
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    stylized, content, style, features['adain_features']
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'val_loss': val_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.get_model().state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoints_dir / f"checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        # Clean up old checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > KEEP_LAST_N_CHECKPOINTS:
            for old_checkpoint in checkpoints[:-KEEP_LAST_N_CHECKPOINTS]:
                old_checkpoint.unlink()
        
        print(f"Checkpoint saved to {epoch_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.get_model().load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Total epochs to train
        """
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Batch size: {len(next(iter(train_loader))['content'])}")
        print(f"Total batches per epoch: {len(train_loader)}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            # Update learning rate
            current_lr = self.update_learning_rate(epoch)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['content_loss'].append(train_metrics['content_loss'])
            self.history['style_loss'].append(train_metrics['style_loss'])
            self.history['identity_loss'].append(train_metrics['identity_loss'])
            self.history['learning_rate'].append(current_lr)
            
            # Logging
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"  Content Loss: {train_metrics['content_loss']:.6f}")
            print(f"  Style Loss: {train_metrics['style_loss']:.6f}")
            print(f"  Identity Loss: {train_metrics['identity_loss']:.6f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            if (epoch + 1) % SAVE_FREQUENCY == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training Complete! Time: {elapsed_time/3600:.2f} hours")
        print("=" * 60)
        
        # Save history
        history_df = pd.DataFrame(self.history)
        history_path = self.logs_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        history_df.to_csv(history_path, index=False)
        print(f"History saved to {history_path}")


def main():
    """Main training script."""
    # Setup
    print("MuseAI Training Script")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Initialize device
    device = get_device()
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        dataloaders = create_dataloaders()
    except FileNotFoundError as e:
        print(f"Error loading preprocessed data: {e}")
        print("Attempting to use raw dataset from datasets/ folder...")
        
        # Use raw style dataset and skip content for now
        from src.utils.data_loader import RawStyleDataset
        raw_dataset = RawStyleDataset()
        
        if len(raw_dataset) == 0:
            print("ERROR: No images found. Please ensure:")
            print("  1. Style images in: datasets/picasso/ and datasets/rembrandt/")
            print("  2. Content faces in: data/content/faces/train/")
            print("\nRun preprocessing first:")
            print("  python -m src.preprocess.style_preprocess")
            print("  python -m src.preprocess.content_preprocess")
            return
    
    # Create model
    print("\nCreating model...")
    model = StyleTransferNetwork(
        use_conditional=True,
        use_improved_decoder=False
    )
    model_sizes = model.get_model_size()
    print("Model Parameters:")
    for name, count in model_sizes.items():
        print(f"  {name}: {count:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        use_amp=USE_AMP,
        use_conditional=True
    )
    
    # Start training
    try:
        trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            num_epochs=NUM_EPOCHS
        )
    except KeyError:
        # If dataloaders not fully loaded, still start with basic setup
        print("\nNote: Full training not possible without preprocessed content data.")
        print("Model is ready. Use:")
        print("  - save_checkpoint() to save the model")
        print("  - load_checkpoint() to load existing models")
    
    print("\nTraining script complete!")


if __name__ == "__main__":
    main()
