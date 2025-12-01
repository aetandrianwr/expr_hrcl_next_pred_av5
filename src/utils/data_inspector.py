"""
Utility to infer dataset parameters from .pk files.
"""

import pickle
import numpy as np
from pathlib import Path


def infer_dataset_parameters(train_file_path):
    """
    Infer dataset parameters from training data file.
    
    Args:
        train_file_path: Path to training .pk file
        
    Returns:
        dict with keys:
            - num_locations: max location ID + 1 (for padding)
            - num_users: max user ID + 1 (for padding)
            - num_weekdays: always 7
            - max_seq_len: maximum sequence length in dataset
    """
    with open(train_file_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected non-empty list in {train_file_path}, got {type(data)}")
    
    # Calculate statistics from all samples
    max_loc = max(max(sample['X']) for sample in data)
    max_user = max(max(sample['user_X']) for sample in data)
    max_seq_len = max(len(sample['X']) for sample in data)
    
    # Also check targets to ensure we capture all locations
    max_target = max(sample['Y'] for sample in data)
    max_loc = max(max_loc, max_target)
    
    params = {
        'num_locations': int(max_loc) + 1,  # +1 for padding (0)
        'num_users': int(max_user) + 1,      # +1 for padding (0)
        'num_weekdays': 7,
        'max_seq_len': int(max_seq_len)
    }
    
    return params
