"""
Dataset and DataLoader for GeoLife trajectory data.
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GeoLifeDataset(Dataset):
    """
    Dataset for GeoLife trajectory sequences.
    
    Each sample contains:
        - X: location sequence (variable length)
        - user_X: user ID for each location
        - weekday_X: weekday for each location
        - start_min_X: start time in minutes from midnight
        - dur_X: duration at each location
        - diff: time gap indicator
        - Y: next location (target)
    """
    
    def __init__(self, data_path, max_seq_len=60):
        """
        Args:
            data_path: Path to pickle file
            max_seq_len: Maximum sequence length (for truncation)
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        loc_seq = sample['X']
        user_seq = sample['user_X']
        weekday_seq = sample['weekday_X']
        start_min_seq = sample['start_min_X']
        dur_seq = sample['dur_X']
        diff_seq = sample['diff']
        target = sample['Y']
        
        # Truncate if too long (keep most recent)
        seq_len = len(loc_seq)
        if seq_len > self.max_seq_len:
            loc_seq = loc_seq[-self.max_seq_len:]
            user_seq = user_seq[-self.max_seq_len:]
            weekday_seq = weekday_seq[-self.max_seq_len:]
            start_min_seq = start_min_seq[-self.max_seq_len:]
            dur_seq = dur_seq[-self.max_seq_len:]
            diff_seq = diff_seq[-self.max_seq_len:]
            seq_len = self.max_seq_len
        
        return {
            'loc_seq': torch.LongTensor(loc_seq),
            'user_seq': torch.LongTensor(user_seq),
            'weekday_seq': torch.LongTensor(weekday_seq),
            'start_min_seq': torch.FloatTensor(start_min_seq),
            'dur_seq': torch.FloatTensor(dur_seq),
            'diff_seq': torch.LongTensor(diff_seq),
            'target': torch.LongTensor([target]),
            'seq_len': seq_len
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Find max length in this batch
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    loc_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    user_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    weekday_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    start_min_seqs = torch.zeros(batch_size, max_len, dtype=torch.float)
    dur_seqs = torch.zeros(batch_size, max_len, dtype=torch.float)
    diff_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in the data
    for i, item in enumerate(batch):
        length = item['seq_len']
        loc_seqs[i, :length] = item['loc_seq']
        user_seqs[i, :length] = item['user_seq']
        weekday_seqs[i, :length] = item['weekday_seq']
        start_min_seqs[i, :length] = item['start_min_seq']
        dur_seqs[i, :length] = item['dur_seq']
        diff_seqs[i, :length] = item['diff_seq']
        targets[i] = item['target']
        seq_lens[i] = length
    
    # Create attention mask (1 for real tokens, 0 for padding)
    mask = torch.arange(max_len).unsqueeze(0) < seq_lens.unsqueeze(1)
    
    return {
        'loc_seq': loc_seqs,
        'user_seq': user_seqs,
        'weekday_seq': weekday_seqs,
        'start_min_seq': start_min_seqs,
        'dur_seq': dur_seqs,
        'diff_seq': diff_seqs,
        'target': targets,
        'mask': mask,
        'seq_len': seq_lens
    }


def get_dataloader(data_path, batch_size, shuffle=True, num_workers=2, max_seq_len=60):
    """
    Create DataLoader for GeoLife dataset.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        max_seq_len: Maximum sequence length
    
    Returns:
        DataLoader instance
    """
    dataset = GeoLifeDataset(data_path, max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
