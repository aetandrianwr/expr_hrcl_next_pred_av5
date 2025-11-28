"""
Configuration for GeoLife next-location prediction.
"""

import torch


class Config:
    """Base configuration for the model and training."""
    
    # Data paths
    data_dir = "data/geolife"
    train_file = "geolife_transformer_7_train.pk"
    val_file = "geolife_transformer_7_validation.pk"
    test_file = "geolife_transformer_7_test.pk"
    
    # Model architecture
    num_locations = 1187  # 1186 max + 1 for padding (0)
    num_users = 46  # 45 max + 1 for padding (0)
    num_weekdays = 7
    
    # Embedding dimensions
    loc_emb_dim = 128
    user_emb_dim = 32
    weekday_emb_dim = 16
    time_emb_dim = 32
    
    # Transformer parameters
    d_model = 256  # Total embedding dimension
    nhead = 8
    num_layers = 4
    dim_feedforward = 512
    dropout = 0.15
    
    # Positional encoding
    max_seq_len = 60
    
    # Training
    batch_size = 128
    num_epochs = 150
    learning_rate = 0.0005
    weight_decay = 1e-4
    grad_clip = 1.0
    
    # Scheduler
    warmup_epochs = 5
    scheduler_patience = 10
    scheduler_factor = 0.5
    min_lr = 1e-6
    
    # Early stopping
    early_stop_patience = 25
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval = 50
    save_dir = "trained_models"
    
    # Random seed
    seed = 42
    
    def __repr__(self):
        attrs = {k: v for k, v in vars(Config).items() 
                if not k.startswith('_') and not callable(v)}
        return '\n'.join(f'{k}: {v}' for k, v in attrs.items())
