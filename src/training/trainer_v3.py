"""
Production-level trainer with YAML configuration support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation.metrics import calculate_correct_total_prediction, get_performance_dict
from sklearn.metrics import f1_score
from utils.logger import ExperimentLogger


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class ProductionTrainer:
    """Production-level trainer with configuration management."""
    
    def __init__(self, model, train_loader, val_loader, config_manager, logger=None):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config_manager: ConfigManager instance
            logger: Optional logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config_manager
        self.device = config_manager.device
        self.logger = logger
        
        self.model.to(self.device)
        
        # Get config values
        lr = self.config.get('training.learning_rate')
        weight_decay = self.config.get('training.weight_decay')
        label_smoothing = self.config.get('training.label_smoothing')
        
        # Optimizer with weight decay only on non-bias/non-norm parameters
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n], 
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n], 
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(param_groups, lr=lr)
        
        # Loss with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # Learning rate scheduler
        scheduler_type = self.config.get('training.scheduler.type')
        if scheduler_type == 'cosine_annealing':
            T_max = self.config.get('training.scheduler.T_max')
            min_lr = self.config.get('training.scheduler.min_lr')
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_max,
                T_mult=1,
                eta_min=min_lr
            )
            self.use_cosine = True
        else:
            patience = self.config.get('training.scheduler.patience')
            factor = self.config.get('training.scheduler.factor')
            min_lr = self.config.get('training.scheduler.min_lr')
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # Monitor loss (lower is better)
                factor=factor,
                patience=patience,
                verbose=True,
                min_lr=min_lr
            )
            self.use_cosine = False
        
        # Training state - use validation loss instead of accuracy
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.start_time = None
    
    def _log(self, message):
        """Log message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def train_epoch(self, epoch):
        """Train for one epoch with progress bar."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        grad_clip = self.config.get('training.grad_clip')
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', leave=False, ncols=100)
        
        for batch in pbar:
            # Move to device
            loc_seq = batch['loc_seq'].to(self.device)
            user_seq = batch['user_seq'].to(self.device)
            weekday_seq = batch['weekday_seq'].to(self.device)
            start_min_seq = batch['start_min_seq'].to(self.device)
            dur_seq = batch['dur_seq'].to(self.device)
            diff_seq = batch['diff_seq'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            logits = self.model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
            
            # Calculate loss
            loss = self.criterion(logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, data_loader, split_name='Val'):
        """Validate the model with progress bar."""
        self.model.eval()
        
        # Initialize metric accumulators
        metrics = {
            "correct@1": 0,
            "correct@3": 0,
            "correct@5": 0,
            "correct@10": 0,
            "rr": 0,
            "ndcg": 0,
            "f1": 0,
            "total": 0
        }
        
        # Lists for F1 score calculation
        true_ls = []
        top1_ls = []
        
        # For loss calculation
        total_val_loss = 0
        num_batches = 0
        
        # Progress bar for validation
        pbar = tqdm(data_loader, desc=f'{split_name:5s}', leave=False, ncols=100)
        
        for batch in pbar:
            # Move to device
            loc_seq = batch['loc_seq'].to(self.device)
            user_seq = batch['user_seq'].to(self.device)
            weekday_seq = batch['weekday_seq'].to(self.device)
            start_min_seq = batch['start_min_seq'].to(self.device)
            dur_seq = batch['dur_seq'].to(self.device)
            diff_seq = batch['diff_seq'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            logits = self.model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
            
            # Calculate loss
            loss = self.criterion(logits, target)
            total_val_loss += loss.item()
            num_batches += 1
            
            # Calculate metrics
            result, batch_true, batch_top1 = calculate_correct_total_prediction(logits, target)
            
            metrics["correct@1"] += result[0]
            metrics["correct@3"] += result[1]
            metrics["correct@5"] += result[2]
            metrics["correct@10"] += result[3]
            metrics["rr"] += result[4]
            metrics["ndcg"] += result[5]
            metrics["total"] += result[6]
            
            # Collect for F1 score
            true_ls.extend(batch_true.tolist())
            if not batch_top1.shape:
                top1_ls.extend([batch_top1.tolist()])
            else:
                top1_ls.extend(batch_top1.tolist())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_val_loss/num_batches:.4f}'})
        
        # Calculate average loss
        avg_val_loss = total_val_loss / num_batches
        
        # Calculate F1 score
        f1 = f1_score(true_ls, top1_ls, average="weighted")
        metrics["f1"] = f1
        
        # Calculate percentages
        perf = get_performance_dict(metrics)
        perf['val_loss'] = avg_val_loss
        
        # Condensed performance display on single line
        self._log(f'{split_name} - Loss: {avg_val_loss:.4f} | Acc@1: {perf["acc@1"]:.2f}% Acc@5: {perf["acc@5"]:.2f}% Acc@10: {perf["acc@10"]:.2f}% | F1: {100*f1:.2f}% MRR: {perf["mrr"]:.2f}% NDCG: {perf["ndcg"]:.2f}%')
        
        return perf
    
    def train(self):
        """Main training loop using validation loss for model selection."""
        num_epochs = self.config.get('training.num_epochs')
        early_stop_patience = self.config.get('training.early_stopping.patience')
        
        self._log(f'\nStarting training on {self.device}')
        self._log(f'Model parameters: {self.model.count_parameters():,}')
        self._log(f'Using validation loss for model selection\n')
        
        self.start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self._log(f'=== Epoch {epoch}/{num_epochs} ===')
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_perf = self.validate(self.val_loader, split_name='Val')
            val_loss = val_perf['val_loss']
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling (use loss for ReduceLROnPlateau)
            if self.use_cosine:
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            # Check for improvement (lower loss is better)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                save_path = self.config.checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_perf['acc@1'],
                    'config': self.config.to_dict()
                }, save_path)
                self._log(f'✓ New best model saved! Val Loss: {val_loss:.4f} (Acc@1: {val_perf["acc@1"]:.2f}%)')
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            self._log(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (best: {self.best_val_loss:.4f} @ epoch {self.best_epoch}) | Time: {epoch_time:.1f}s')
            self._log(f'Epochs without improvement: {self.epochs_without_improvement}/{early_stop_patience}\n')
            
            # Early stopping
            if self.epochs_without_improvement >= early_stop_patience:
                self._log(f'Early stopping triggered after {epoch} epochs')
                break
        
        total_time = time.time() - self.start_time
        
        self._log(f'\nTraining completed!')
        self._log(f'Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}')
        self._log(f'Total training time: {total_time:.2f}s')
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch,
            'training_time': total_time
        }
    
    def load_best_model(self):
        """Load the best saved model."""
        save_path = self.config.checkpoint_dir / 'best_model.pt'
        if save_path.exists():
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            val_loss = checkpoint.get('val_loss', 'N/A')
            val_acc = checkpoint.get('val_acc', 'N/A')
            if isinstance(val_loss, float):
                self._log(f'Loaded best model from epoch {checkpoint["epoch"]} (Val Loss: {val_loss:.4f}, Acc@1: {val_acc:.2f}%)')
            else:
                self._log(f'Loaded best model from epoch {checkpoint["epoch"]}')
            return True
        return False
