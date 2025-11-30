"""
Main training script for GeoLife next-location prediction.
"""

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import Config
from data.dataset import get_dataloader
from models.history_centric import HistoryCentricModel
from training.trainer import TrainerV2


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load config
    config = Config()
    
    # Set random seed
    set_seed(config.seed)
    
    print("=" * 80)
    print("GeoLife Next-Location Prediction - Hierarchical Transformer")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Num epochs: {config.num_epochs}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Num heads: {config.nhead}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Dropout: {config.dropout}")
    print()
    
    # Create data loaders
    print("Loading data...")
    train_loader = get_dataloader(
        os.path.join(config.data_dir, config.train_file),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        max_seq_len=config.max_seq_len
    )
    val_loader = get_dataloader(
        os.path.join(config.data_dir, config.val_file),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        max_seq_len=config.max_seq_len
    )
    test_loader = get_dataloader(
        os.path.join(config.data_dir, config.test_file),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        max_seq_len=config.max_seq_len
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = HistoryCentricModel(config)
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    if num_params >= 500000:
        print(f"WARNING: Model has {num_params:,} parameters (limit is 500K)")
        print(f"Exceeded by: {num_params - 500000:,}")
    else:
        print(f"✓ Model is within budget (remaining: {500000 - num_params:,})")
    print()
    
    # Create trainer
    trainer = TrainerV2(model, train_loader, val_loader, config)
    
    # Train
    best_val_acc = trainer.train()
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating best model on test set...")
    print("=" * 80)
    trainer.load_best_model()
    test_perf = trainer.validate(test_loader, split_name='Test')
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Validation Acc@1: {best_val_acc:.2f}%")
    print(f"Test Acc@1: {test_perf['acc@1']:.2f}%")
    print(f"Test Acc@5: {test_perf['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"Test F1: {100 * test_perf['f1']:.2f}%")
    print(f"Test MRR: {test_perf['mrr']:.2f}%")
    print(f"Test NDCG: {test_perf['ndcg']:.2f}%")
    print("=" * 80)
    
    # Check if target is met
    if test_perf['acc@1'] >= 50.0:
        print("\n✓ SUCCESS: Test Acc@1 >= 50% achieved!")
    else:
        print(f"\n✗ Test Acc@1 {test_perf['acc@1']:.2f}% is below 50% target")
        print("  Model needs further optimization")
    
    return test_perf


if __name__ == "__main__":
    main()
