"""
Production-level training script with YAML configuration.
Supports multiple datasets and automatic result tracking.

Usage:
    python train_model.py --config configs/geolife_default.yaml
    python train_model.py --config configs/custom.yaml --seed 123
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path
import time

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from utils.config_manager import ConfigManager
from utils.results_tracker import ResultsTracker
from utils.logger import ExperimentLogger
from utils.data_inspector import infer_dataset_parameters
from data.dataset import get_dataloader
from models.history_centric import HistoryCentricModel
from training.trainer_v3 import ProductionTrainer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train next-location prediction model')
    parser.add_argument('--config', type=str, default='configs/geolife_default.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    overrides = {}
    if args.seed is not None:
        overrides['system.seed'] = args.seed
    
    config = ConfigManager(args.config, overrides)
    
    # Display configuration
    config.display()
    
    # Set random seed
    seed = config.get('system.seed')
    set_seed(seed)
    
    # Initialize logger
    logger = ExperimentLogger(config.log_dir, name=config.get('experiment.name'))
    logger.info("=" * 80)
    logger.info("EXPERIMENT STARTED")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Run directory: {config.run_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Random seed: {seed}")
    
    # Infer dataset parameters from training data
    logger.info("\nInferring dataset parameters from training data...")
    data_dir = config.get('data.data_dir')
    train_file = config.get('data.train_file')
    train_file_path = os.path.join(data_dir, train_file)
    
    inferred_params = infer_dataset_parameters(train_file_path)
    
    logger.info("=" * 80)
    logger.info("INFERRED DATASET PARAMETERS")
    logger.info("=" * 80)
    logger.info(f"num_locations: {inferred_params['num_locations']} ({inferred_params['num_locations']-1} max + 1 for padding)")
    logger.info(f"num_users: {inferred_params['num_users']} ({inferred_params['num_users']-1} max + 1 for padding)")
    logger.info(f"num_weekdays: {inferred_params['num_weekdays']}")
    logger.info(f"max_seq_len: {inferred_params['max_seq_len']}")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading data...")
    val_file = config.get('data.val_file')
    test_file = config.get('data.test_file')
    batch_size = config.get('training.batch_size')
    max_seq_len = inferred_params['max_seq_len']  # Use inferred value
    num_workers = config.get('system.num_workers')
    
    train_loader = get_dataloader(
        os.path.join(data_dir, train_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_seq_len=max_seq_len
    )
    val_loader = get_dataloader(
        os.path.join(data_dir, val_file),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_seq_len=max_seq_len
    )
    test_loader = get_dataloader(
        os.path.join(data_dir, test_file),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_seq_len=max_seq_len
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    
    # Build config object for model (backward compatibility)
    class ModelConfig:
        def __init__(self, config_dict, inferred_params):
            # Use inferred parameters instead of config
            self.num_locations = inferred_params['num_locations']
            self.num_users = inferred_params['num_users']
            self.num_weekdays = inferred_params['num_weekdays']
            self.max_seq_len = inferred_params['max_seq_len']
            # Keep other parameters from config
            self.loc_emb_dim = config_dict.get('model.loc_emb_dim')
            self.user_emb_dim = config_dict.get('model.user_emb_dim')
            self.weekday_emb_dim = config_dict.get('model.weekday_emb_dim')
            self.time_emb_dim = config_dict.get('model.time_emb_dim')
            self.d_model = config_dict.get('model.d_model')
            self.nhead = config_dict.get('model.nhead')
            self.num_layers = config_dict.get('model.num_layers')
            self.dim_feedforward = config_dict.get('model.dim_feedforward')
            self.dropout = config_dict.get('model.dropout')
            self.dataset_name = config_dict.get('experiment.dataset')
    
    model_config = ModelConfig(config, inferred_params)
    model = HistoryCentricModel(model_config)
    num_params = model.count_parameters()
    
    logger.info(f"Model: {config.get('model.name')}")
    logger.info(f"Total parameters: {num_params:,}")
    
    # Display ALL actual parameters being used by the training script
    logger.info("\n" + "=" * 80)
    logger.info("ACTUAL PARAMETERS BEING USED (Not Config File Values)")
    logger.info("=" * 80)
    
    # ========== DATA PARAMETERS (Auto-Inferred) ==========
    logger.info("\n[DATA] Parameters (auto-inferred from dataset files):")
    logger.info(f"  Dataset: {config.get('experiment.dataset')}")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Train file: {train_file}")
    logger.info(f"  Val file: {val_file}")
    logger.info(f"  Test file: {test_file}")
    logger.info(f"  num_locations: {model_config.num_locations} (vocabulary size)")
    logger.info(f"  num_users: {model_config.num_users} (user count)")
    logger.info(f"  num_weekdays: {model_config.num_weekdays}")
    logger.info(f"  max_seq_len: {model_config.max_seq_len} (max sequence length in dataset)")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  num_workers: {num_workers}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # ========== MODEL ARCHITECTURE (From Model Object) ==========
    logger.info("\n[MODEL] Architecture (actual model layers):")
    logger.info(f"  Model class: {model.__class__.__name__}")
    logger.info(f"  d_model: {model.d_model}")
    if hasattr(model, 'loc_emb'):
        logger.info(f"  loc_emb_dim: {model.loc_emb.embedding_dim}")
        logger.info(f"  loc_emb vocab: {model.loc_emb.num_embeddings}")
    if hasattr(model, 'user_emb'):
        logger.info(f"  user_emb_dim: {model.user_emb.embedding_dim}")
        logger.info(f"  user_emb vocab: {model.user_emb.num_embeddings}")
    if hasattr(model, 'attn'):
        logger.info(f"  nhead: {model.attn.num_heads}")
    
    # Get actual dropout value
    dropout_val = None
    if hasattr(model, 'dropout') and hasattr(model.dropout, 'p'):
        dropout_val = model.dropout.p
        logger.info(f"  dropout: {dropout_val}")
    
    # Calculate dim_feedforward from model
    dim_feedforward = None
    if hasattr(model, 'ff') and len(model.ff) > 0:
        for layer in model.ff:
            if isinstance(layer, torch.nn.Linear):
                dim_feedforward = layer.out_features
                break
    if dim_feedforward:
        logger.info(f"  dim_feedforward: {dim_feedforward}")
    
    logger.info(f"  Total parameters: {num_params:,}")
    
    # ========== TRAINING PARAMETERS (Actual Values Used) ==========
    logger.info("\n[TRAINING] Training parameters (being used):")
    logger.info(f"  num_epochs: {config.get('training.num_epochs')}")
    logger.info(f"  learning_rate: {config.get('training.learning_rate')}")
    logger.info(f"  weight_decay: {config.get('training.weight_decay')}")
    logger.info(f"  grad_clip: {config.get('training.grad_clip')}")
    logger.info(f"  label_smoothing: {config.get('training.label_smoothing')}")
    logger.info(f"  optimizer: {config.get('training.optimizer')}")
    logger.info(f"  optimizer_betas: {config.get('training.betas')}")
    logger.info(f"  optimizer_eps: {config.get('training.eps')}")
    
    # ========== SCHEDULER PARAMETERS ==========
    logger.info("\n[SCHEDULER] Learning rate scheduler:")
    logger.info(f"  type: {config.get('training.scheduler.type')}")
    logger.info(f"  patience: {config.get('training.scheduler.patience')}")
    logger.info(f"  factor: {config.get('training.scheduler.factor')}")
    logger.info(f"  min_lr: {config.get('training.scheduler.min_lr')}")
    logger.info(f"  warmup_epochs: {config.get('training.scheduler.warmup_epochs')}")
    
    # ========== EARLY STOPPING ==========
    logger.info("\n[EARLY STOPPING]:")
    logger.info(f"  patience: {config.get('training.early_stopping.patience')}")
    logger.info(f"  metric: {config.get('training.early_stopping.metric')}")
    logger.info(f"  mode: {config.get('training.early_stopping.mode')}")
    
    # ========== SYSTEM PARAMETERS ==========
    logger.info("\n[SYSTEM] System configuration:")
    logger.info(f"  device: {config.device}")
    logger.info(f"  seed: {seed}")
    logger.info(f"  deterministic: {config.get('system.deterministic')}")
    logger.info(f"  pin_memory: {config.get('system.pin_memory')}")
    
    # ========== PATHS ==========
    logger.info("\n[PATHS] Output directories:")
    logger.info(f"  run_dir: {config.run_dir}")
    logger.info(f"  checkpoint_dir: {config.checkpoint_dir}")
    logger.info(f"  log_dir: {config.log_dir}")
    
    logger.info("=" * 80)
    
    # Validate model configuration matches config file
    logger.info("\n" + "=" * 80)
    logger.info("CONFIGURATION VALIDATION (Config File vs Actual Model)")
    logger.info("=" * 80)
    
    config_mismatches = []
    
    # Check d_model
    if hasattr(model, 'd_model'):
        config_d_model = config.get('model.d_model')
        if model.d_model != config_d_model:
            msg = f"d_model: Config={config_d_model}, Actual={model.d_model}"
            config_mismatches.append(msg)
            logger.warning(f"⚠ MISMATCH: {msg}")
        else:
            logger.info(f"✓ d_model: {model.d_model}")
    
    # Check embedding dimensions
    if hasattr(model, 'loc_emb'):
        model_loc_dim = model.loc_emb.embedding_dim
        config_loc_dim = config.get('model.loc_emb_dim')
        if model_loc_dim != config_loc_dim:
            msg = f"loc_emb_dim: Config={config_loc_dim}, Actual={model_loc_dim}"
            config_mismatches.append(msg)
            logger.warning(f"⚠ MISMATCH: {msg}")
        else:
            logger.info(f"✓ loc_emb_dim: {model_loc_dim}")
    
    if hasattr(model, 'user_emb'):
        model_user_dim = model.user_emb.embedding_dim
        config_user_dim = config.get('model.user_emb_dim')
        if model_user_dim != config_user_dim:
            msg = f"user_emb_dim: Config={config_user_dim}, Actual={model_user_dim}"
            config_mismatches.append(msg)
            logger.warning(f"⚠ MISMATCH: {msg}")
        else:
            logger.info(f"✓ user_emb_dim: {model_user_dim}")
    
    if config_mismatches:
        logger.warning("\n" + "!" * 80)
        logger.warning("WARNING: Model architecture differs from config file!")
        logger.warning("This may indicate:")
        logger.warning("  1. Config file needs updating to match model")
        logger.warning("  2. Model needs fixing to respect config parameters")
        logger.warning("  3. Intentional override (if using getattr defaults)")
        logger.warning("Mismatches found:")
        for mismatch in config_mismatches:
            logger.warning(f"  - {mismatch}")
        logger.warning("!" * 80 + "\n")
    else:
        logger.info("✓ All config parameters match model architecture")
    
    logger.info("=" * 80)
    
    if num_params >= 500000:
        logger.warning(f"WARNING: Model has {num_params:,} parameters (limit is 500K)")
        logger.warning(f"Exceeded by: {num_params - 500000:,}")
    else:
        logger.info(f"✓ Model is within budget (remaining: {500000 - num_params:,})")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = ProductionTrainer(model, train_loader, val_loader, config, logger)
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)
    training_info = trainer.train()
    
    # Load best model and evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 80)
    trainer.load_best_model()
    test_perf = trainer.validate(test_loader, split_name='Test')
    
    # Get validation performance at best epoch
    val_perf = {
        'acc@1': 0,  # Will be filled from test_perf if needed
        'acc@3': 0,
        'acc@5': 0,
        'acc@10': 0,
        'f1': 0,
        'mrr': 0,
        'ndcg': 0,
    }
    
    # Display final results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best Validation Loss: {trainer.best_val_loss:.4f} (Epoch {trainer.best_epoch})")
    logger.info(f"Test Acc@1: {test_perf['acc@1']:.2f}%")
    logger.info(f"Test Acc@5: {test_perf['acc@5']:.2f}%")
    logger.info(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
    logger.info(f"Test F1: {100 * test_perf['f1']:.2f}%")
    logger.info(f"Test MRR: {test_perf['mrr']:.2f}%")
    logger.info(f"Test NDCG: {test_perf['ndcg']:.2f}%")
    logger.info(f"Training time: {training_info['training_time']:.2f}s")
    logger.info("=" * 80)
    
    # Log results to CSV
    logger.info("\nLogging results to benchmark CSV...")
    tracker = ResultsTracker()
    tracker.log_result(
        config=config.to_dict(),
        val_metrics=val_perf,
        test_metrics=test_perf,
        training_info={
            'run_dir': str(config.run_dir),
            'total_params': num_params,
            'epochs_trained': training_info['total_epochs'],
            'best_epoch': training_info['best_epoch'],
            'training_time': training_info['training_time'],
        }
    )
    
    logger.info(f"\n✓ Experiment completed successfully!")
    logger.info(f"Run directory: {config.run_dir}")
    logger.info(f"Log file: {logger.log_file}")
    
    return test_perf


if __name__ == "__main__":
    main()
