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
                mode='max',
                factor=factor,
                patience=patience,
                verbose=True,
                min_lr=min_lr
            )
            self.use_cosine = False
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_accs = []
        self.start_time = None
    
    def _log(self, message):
        """Log message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        log_interval = self.config.get('logging.log_interval')
        grad_clip = self.config.get('training.grad_clip')
        
        for batch_idx, batch in enumerate(self.train_loader):
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
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                self._log(f'Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] Loss: {avg_loss:.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, data_loader, split_name='Val'):
        """Validate the model."""
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
        
        for batch in data_loader:
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
        
        # Calculate F1 score
        f1 = f1_score(true_ls, top1_ls, average="weighted")
        metrics["f1"] = f1
        
        # Calculate percentages
        perf = get_performance_dict(metrics)
        
        self._log(f'\n{split_name} Performance:')
        self._log(f'  Acc@1:  {perf["acc@1"]:.2f}%')
        self._log(f'  Acc@5:  {perf["acc@5"]:.2f}%')
        self._log(f'  Acc@10: {perf["acc@10"]:.2f}%')
        self._log(f'  F1:     {100 * f1:.2f}%')
        self._log(f'  MRR:    {perf["mrr"]:.2f}%')
        self._log(f'  NDCG:   {perf["ndcg"]:.2f}%\n')
        
        return perf
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('training.num_epochs')
        early_stop_patience = self.config.get('training.early_stopping.patience')
        
        self._log(f'\nStarting training on {self.device}')
        self._log(f'Model parameters: {self.model.count_parameters():,}\n')
        
        self.start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self._log(f'=== Epoch {epoch}/{num_epochs} ===')
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_perf = self.validate(self.val_loader, split_name='Val')
            val_acc = val_perf['acc@1']
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            if self.use_cosine:
                self.scheduler.step()
            else:
                self.scheduler.step(val_acc)
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                save_path = self.config.checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config.to_dict()
                }, save_path)
                self._log(f'✓ New best model saved! Val Acc@1: {val_acc:.2f}%')
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            self._log(f'Epoch time: {epoch_time:.2f}s')
            self._log(f'Best Val Acc@1: {self.best_val_acc:.2f}% (epoch {self.best_epoch})')
            self._log(f'Epochs without improvement: {self.epochs_without_improvement}/{early_stop_patience}\n')
            
            # Early stopping
            if self.epochs_without_improvement >= early_stop_patience:
                self._log(f'Early stopping triggered after {epoch} epochs')
                break
        
        total_time = time.time() - self.start_time
        
        self._log(f'\nTraining completed!')
        self._log(f'Best validation Acc@1: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
        self._log(f'Total training time: {total_time:.2f}s')
        
        return {
            'best_val_acc': self.best_val_acc,
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
            self._log(f'Loaded best model from epoch {checkpoint["epoch"]} with val acc {checkpoint["val_acc"]:.2f}%')
            return True
        return False
