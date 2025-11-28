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
    
    # Embedding dimensions - optimized for hybrid model
    loc_emb_dim = 64
    user_emb_dim = 16
    weekday_emb_dim = 4
    time_emb_dim = 8
    
    # Transformer parameters - lightweight for hybrid approach
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.3  # Stronger dropout for hybrid model
    
    # Positional encoding
    max_seq_len = 60
    
    # Training - optimized with stronger regularization
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.0008
    weight_decay = 1e-4
    grad_clip = 0.5
    label_smoothing = 0.08
    
    # Scheduler
    warmup_epochs = 10
    scheduler_patience = 15
    scheduler_factor = 0.6
    min_lr = 1e-6
    use_cosine_annealing = True
    T_max = 50
    
    # Early stopping
    early_stop_patience = 35
    
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
