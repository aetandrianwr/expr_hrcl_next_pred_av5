"""
Training script for Advanced History Transformer with frequency analysis
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import pickle
from collections import Counter
import numpy as np

from utils.config_manager import ConfigManager
from utils.data_inspector import infer_dataset_parameters
from models.advanced_history_transformer import AdvancedHistoryTransformer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm


class LocationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'loc_seq': torch.LongTensor(sample['X']),
            'user_seq': torch.LongTensor(sample['user_X']),
            'weekday_seq': torch.LongTensor(sample['weekday_X']),
            'start_min_seq': torch.FloatTensor(sample['start_min_X']),
            'dur_seq': torch.FloatTensor(sample['dur_X']),
            'diff_seq': torch.LongTensor(sample['diff']),
            'target': torch.LongTensor([sample['Y']])[0]
        }


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    max_len = max([item['loc_seq'].size(0) for item in batch])
    
    batch_size = len(batch)
    loc_seq = torch.zeros(batch_size, max_len, dtype=torch.long)
    user_seq = torch.zeros(batch_size, max_len, dtype=torch.long)
    weekday_seq = torch.zeros(batch_size, max_len, dtype=torch.long)
    start_min_seq = torch.zeros(batch_size, max_len, dtype=torch.float)
    dur_seq = torch.zeros(batch_size, max_len, dtype=torch.float)
    diff_seq = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['loc_seq'].size(0)
        loc_seq[i, :seq_len] = item['loc_seq']
        user_seq[i, :seq_len] = item['user_seq']
        weekday_seq[i, :seq_len] = item['weekday_seq']
        start_min_seq[i, :seq_len] = item['start_min_seq']
        dur_seq[i, :seq_len] = item['dur_seq']
        diff_seq[i, :seq_len] = item['diff_seq']
        mask[i, :seq_len] = True
        targets[i] = item['target']
    
    return {
        'loc_seq': loc_seq,
        'user_seq': user_seq,
        'weekday_seq': weekday_seq,
        'start_min_seq': start_min_seq,
        'dur_seq': dur_seq,
        'diff_seq': diff_seq,
        'mask': mask,
        'targets': targets
    }


def compute_location_frequencies(train_data):
    """Compute location visit frequencies."""
    loc_counter = Counter()
    
    for sample in train_data:
        loc_counter.update(sample['X'].tolist())
        loc_counter[sample['Y']] += 1
    
    return loc_counter


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
        loss = criterion(logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), 100 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        
        # Top-k accuracy
        _, pred_top5 = logits.topk(5, dim=1)
        _, pred_top10 = logits.topk(10, dim=1)
        
        correct_at_1 += (pred_top5[:, 0] == targets).sum().item()
        correct_at_5 += (pred_top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct_at_10 += (pred_top10 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({'acc@1': f'{100*correct_at_1/total:.2f}%'})
    
    return {
        'loss': total_loss / len(dataloader),
        'acc@1': 100 * correct_at_1 / total,
        'acc@5': 100 * correct_at_5 / total,
        'acc@10': 100 * correct_at_10 / total
    }


def main():
    # Load data
    print("Loading data...")
    with open('data/diy/diy_transformer_7_train.pk', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/diy/diy_transformer_7_validation.pk', 'rb') as f:
        val_data = pickle.load(f)
    with open('data/diy/diy_transformer_7_test.pk', 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Compute location frequencies
    print("Computing location frequencies...")
    loc_counter = compute_location_frequencies(train_data)
    
    # Get top-K locations
    top_k_locs = [loc for loc, _ in loc_counter.most_common(2500)]
    location_freqs = np.ones(15584)
    for loc, count in loc_counter.items():
        if loc < len(location_freqs):
            location_freqs[loc] = count
    
    print(f"Top-2500 locations cover {sum([loc_counter[loc] for loc in top_k_locs]) / sum(loc_counter.values()) * 100:.2f}% of visits")
    
    # Create model
    print("Creating model...")
    
    class ModelConfig:
        num_locations = 15584
        num_users = 803
    
    config = ModelConfig()
    model = AdvancedHistoryTransformer(config)
    model.set_frequency_mapping(top_k_locs, location_freqs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    
    # Create dataloaders
    train_dataset = LocationDataset(train_data)
    val_dataset = LocationDataset(val_data)
    test_dataset = LocationDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_acc = 0
    patience = 10
    no_improve = 0
    
    print("\nStarting training...")
    for epoch in range(1, 121):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/120")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc@1: {val_metrics['acc@1']:.2f}%")
        print(f"Val Acc@5: {val_metrics['acc@5']:.2f}%, Val Acc@10: {val_metrics['acc@10']:.2f}%")
        
        scheduler.step(val_metrics['acc@1'])
        
        # Save best model
        if val_metrics['acc@1'] > best_val_acc:
            best_val_acc = val_metrics['acc@1']
            torch.save(model.state_dict(), 'best_advanced_model.pt')
            print(f"✓ New best model saved! Val Acc@1: {best_val_acc:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
        
        if no_improve >= patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "="*60)
    print("Evaluating best model on test set...")
    print("="*60)
    
    model.load_state_dict(torch.load('best_advanced_model.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Acc@1: {test_metrics['acc@1']:.2f}%")
    print(f"Test Acc@5: {test_metrics['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_metrics['acc@10']:.2f}%")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
