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
    
    # Embedding dimensions - reduced for parameter efficiency
    loc_emb_dim = 96
    user_emb_dim = 24
    weekday_emb_dim = 8
    time_emb_dim = 16
    
    # Transformer parameters - optimized for < 500K params
    d_model = 192  # Total embedding dimension
    nhead = 6
    num_layers = 3
    dim_feedforward = 384
    dropout = 0.2
    
    # Positional encoding
    max_seq_len = 60
    
    # Training - optimized for better convergence
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.001
    weight_decay = 5e-5
    grad_clip = 1.0
    label_smoothing = 0.05
    
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
