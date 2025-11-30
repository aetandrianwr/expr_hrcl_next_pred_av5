"""
Training utilities and trainer class with enhanced optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation.metrics import calculate_correct_total_prediction, get_performance_dict
from sklearn.metrics import f1_score


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


class FocalLoss(nn.Module):
    """
    Focal loss to handle class imbalance.
    Focuses on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class TrainerV2:
    """Enhanced trainer with better optimization."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device
        
        self.model.to(self.device)
        
        # Optimizer with weight decay only on non-bias/non-norm parameters
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n], 
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n], 
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(param_groups, lr=config.learning_rate)
        
        # Loss with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(smoothing=getattr(config, 'label_smoothing', 0.05))
        
        # Learning rate scheduler
        if getattr(config, 'use_cosine_annealing', False):
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.T_max,
                T_mult=1,
                eta_min=config.min_lr
            )
            self.use_cosine = True
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                verbose=True,
                min_lr=config.min_lr
            )
            self.use_cosine = False
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_accs = []
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] Loss: {avg_loss:.4f}')
        
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
        
        print(f'\n{split_name} Performance:')
        print(f'  Acc@1:  {perf["acc@1"]:.2f}%')
        print(f'  Acc@5:  {perf["acc@5"]:.2f}%')
        print(f'  Acc@10: {perf["acc@10"]:.2f}%')
        print(f'  F1:     {100 * f1:.2f}%')
        print(f'  MRR:    {perf["mrr"]:.2f}%')
        print(f'  NDCG:   {perf["ndcg"]:.2f}%\n')
        
        return perf
    
    def train(self):
        """Main training loop."""
        print(f'\nStarting training on {self.device}')
        print(f'Model parameters: {self.model.count_parameters():,}\n')
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f'=== Epoch {epoch}/{self.config.num_epochs} ===')
            start_time = time.time()
            
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
                save_path = os.path.join(self.config.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, save_path)
                print(f'✓ New best model saved! Val Acc@1: {val_acc:.2f}%')
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - start_time
            print(f'Epoch time: {epoch_time:.2f}s')
            print(f'Best Val Acc@1: {self.best_val_acc:.2f}% (epoch {self.best_epoch})')
            print(f'Epochs without improvement: {self.epochs_without_improvement}/{self.config.early_stop_patience}\n')
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stop_patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
        
        print(f'\nTraining completed!')
        print(f'Best validation Acc@1: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
        
        return self.best_val_acc
    
    def load_best_model(self):
        """Load the best saved model."""
        save_path = os.path.join(self.config.save_dir, 'best_model.pt')
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded best model from epoch {checkpoint["epoch"]} with val acc {checkpoint["val_acc"]:.2f}%')
            return True
        return False
