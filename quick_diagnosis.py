"""
Quick diagnostic script to identify the key issue with DIY dataset.
We'll test various hypotheses about why history module doesn't work.
"""

import os
import sys
import torch
import pickle
import numpy as np
import math

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from utils.config_manager import ConfigManager
from utils.data_inspector import infer_dataset_parameters
from data.dataset import get_dataloader
from models.history_centric import HistoryCentricModel

def analyze_history_effectiveness(config_path, num_batches=50):
    """
    Test if history scores correlate with ground truth.
    If history module should work, targets should have high history scores.
    """
    
    config = ConfigManager(config_path, {})
    
    # Load test data
    data_dir = config.get('data.data_dir')
    test_file = config.get('data.test_file')
    test_file_path = os.path.join(data_dir, test_file)
    
    inferred_params = infer_dataset_parameters(test_file_path)
    
    test_loader = get_dataloader(
        test_file_path,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        max_seq_len=inferred_params['max_seq_len']
    )
    
    # Create model (untrained - just for history scoring)
    class ModelConfig:
        def __init__(self, config_dict, inferred_params):
            self.num_locations = inferred_params['num_locations']
            self.num_users = inferred_params['num_users']
            self.num_weekdays = inferred_params['num_weekdays']
            self.max_seq_len = inferred_params['max_seq_len']
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
    model.eval()
    
    print("\n" + "=" * 80)
    print(f"ANALYZING: {config.get('experiment.dataset').upper()}")
    print("=" * 80)
    
    print(f"\nDataset parameters:")
    print(f"  num_locations: {inferred_params['num_locations']}")
    print(f"  num_users: {inferred_params['num_users']}")
    print(f"  max_seq_len: {inferred_params['max_seq_len']}")
    
    print(f"\nHistory module parameters (initial):")
    print(f"  recency_decay: {model.recency_decay.item():.4f}")
    print(f"  freq_weight: {model.freq_weight.item():.4f}")
    print(f"  history_scale: {model.history_scale.item():.4f}")
    print(f"  model_weight: {model.model_weight.item():.4f}")
    
    # Collect statistics
    targets_in_history = []
    target_recency_ranks = []
    target_freq_ranks = []
    target_history_scores = []
    max_history_scores = []
    sequence_lengths = []
    unique_locs_in_seq = []
    
    device = torch.device('cpu')  # Use CPU for quick analysis
    model = model.to(device)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break
            
            loc_seq = batch['loc_seq'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            # Compute history scores
            history_scores = model.compute_history_scores(loc_seq, mask)
            
            batch_size = loc_seq.shape[0]
            for j in range(batch_size):
                target = targets[j].item()
                seq = loc_seq[j].cpu().numpy()
                mask_j = mask[j].cpu().numpy()
                valid_seq = seq[mask_j]
                
                sequence_lengths.append(len(valid_seq))
                unique_locs_in_seq.append(len(set(valid_seq)))
                
                # Check if target in history
                in_hist = int(target in valid_seq)
                targets_in_history.append(in_hist)
                
                # Get history score for target
                hist_scores = history_scores[j].cpu().numpy()
                target_score = hist_scores[target]
                max_score = hist_scores.max()
                
                target_history_scores.append(target_score)
                max_history_scores.append(max_score)
                
                # Rank of target by history score
                rank = (hist_scores > target_score).sum() + 1
                target_recency_ranks.append(rank)
    
    # Report statistics
    print(f"\nStatistics from {len(targets_in_history)} samples:")
    print(f"\nSequence characteristics:")
    print(f"  Avg sequence length: {np.mean(sequence_lengths):.2f}")
    print(f"  Avg unique locations: {np.mean(unique_locs_in_seq):.2f}")
    print(f"  Repeat rate: {1 - np.mean(unique_locs_in_seq)/np.mean(sequence_lengths):.2%}")
    
    print(f"\nTarget coverage:")
    print(f"  Targets in history: {np.mean(targets_in_history):.2%}")
    
    print(f"\nHistory score analysis:")
    print(f"  Target history score - Mean: {np.mean(target_history_scores):.4f}, Std: {np.std(target_history_scores):.4f}")
    print(f"  Max history score    - Mean: {np.mean(max_history_scores):.4f}, Std: {np.std(max_history_scores):.4f}")
    print(f"  Score ratio (target/max): {np.mean(np.array(target_history_scores) / np.array(max_history_scores)):.2%}")
    
    print(f"\nRanking by history:")
    print(f"  Avg rank of target: {np.mean(target_recency_ranks):.1f}")
    print(f"  Targets in top-1: {np.mean(np.array(target_recency_ranks) == 1):.2%}")
    print(f"  Targets in top-5: {np.mean(np.array(target_recency_ranks) <= 5):.2%}")
    print(f"  Targets in top-10: {np.mean(np.array(target_recency_ranks) <= 10):.2%}")
    
    # Estimate upper bound of history-only model
    history_only_acc1 = np.mean(np.array(target_recency_ranks) == 1)
    print(f"\n⚠ CRITICAL: History-only Acc@1 upper bound: {history_only_acc1:.2%}")
    print(f"   This is the MAXIMUM the history module alone can achieve.")
    
    # Check if history scores are discriminative
    targets_in_hist = np.array(targets_in_history) == 1
    if targets_in_hist.sum() > 0:
        hist_scores_for_targets_in_hist = np.array(target_history_scores)[targets_in_hist]
        max_scores_for_targets_in_hist = np.array(max_history_scores)[targets_in_hist]
        
        print(f"\nFor targets that ARE in history ({targets_in_hist.sum()} samples):")
        print(f"  Avg target score: {hist_scores_for_targets_in_hist.mean():.4f}")
        print(f"  Avg max score: {max_scores_for_targets_in_hist.mean():.4f}")
        print(f"  Ratio: {(hist_scores_for_targets_in_hist / max_scores_for_targets_in_hist).mean():.2%}")
    
    return {
        'targets_in_history_pct': np.mean(targets_in_history),
        'history_only_acc1': history_only_acc1,
        'avg_seq_len': np.mean(sequence_lengths),
        'avg_unique_locs': np.mean(unique_locs_in_seq),
    }

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HISTORY MODULE EFFECTIVENESS DIAGNOSIS")
    print("=" * 80)
    
    print("\nThis script tests if the history module CAN work on each dataset.")
    print("If targets are rarely in history or poorly ranked, history won't help.\n")
    
    diy_stats = analyze_history_effectiveness('configs/diy.yaml', num_batches=100)
    
    print("\n" + "=" * 80)
    
    geo_stats = analyze_history_effectiveness('configs/geolife_default.yaml', num_batches=100)
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    print(f"\nDIY vs Geolife:")
    print(f"  Targets in history: DIY={diy_stats['targets_in_history_pct']:.2%}, Geo={geo_stats['targets_in_history_pct']:.2%}")
    print(f"  History-only Acc@1: DIY={diy_stats['history_only_acc1']:.2%}, Geo={geo_stats['history_only_acc1']:.2%}")
    print(f"  Avg seq length: DIY={diy_stats['avg_seq_len']:.1f}, Geo={geo_stats['avg_seq_len']:.1f}")
    print(f"  Avg unique locs: DIY={diy_stats['avg_unique_locs']:.1f}, Geo={geo_stats['avg_unique_locs']:.1f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if diy_stats['history_only_acc1'] < 0.40:
        print("\n⚠ PROBLEM IDENTIFIED:")
        print(f"  DIY history-only Acc@1 is only {diy_stats['history_only_acc1']:.2%}")
        print(f"  This means the history module CANNOT achieve >40% alone!")
        print(f"  The issue is NOT in the implementation, but in the APPROACH.")
        print(f"\n  SOLUTION: The model needs to rely MORE on learned patterns,")
        print(f"            not history. Increase model_weight parameter significantly.")
    elif diy_stats['history_only_acc1'] >= 0.60:
        print("\n✓ History module HAS potential:")
        print(f"  DIY history-only could achieve {diy_stats['history_only_acc1']:.2%}")
        print(f"  The issue is likely in:")
        print(f"    1. Parameter initialization")
        print(f"    2. History scoring formula")
        print(f"    3. Combination with learned model")
    else:
        print("\n⚠ Mixed results:")
        print(f"  DIY history-only can achieve {diy_stats['history_only_acc1']:.2%}")
        print(f"  This is moderate - history helps but isn't enough alone.")
        print(f"  Need better balance between history and learned components.")
