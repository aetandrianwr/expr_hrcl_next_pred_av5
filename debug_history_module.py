"""
Debug script to check if history module is contributing to predictions.
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from utils.config_manager import ConfigManager
from utils.data_inspector import infer_dataset_parameters
from data.dataset import get_dataloader
from models.history_centric import HistoryCentricModel

def analyze_history_contribution(config_path, checkpoint_path):
    """Analyze how much the history module contributes to predictions."""
    
    # Load configuration
    config = ConfigManager(config_path, {})
    
    # Infer dataset parameters
    data_dir = config.get('data.data_dir')
    test_file = config.get('data.test_file')
    test_file_path = os.path.join(data_dir, test_file)
    
    inferred_params = infer_dataset_parameters(test_file_path)
    
    # Load test data (small batch)
    test_loader = get_dataloader(
        test_file_path,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        max_seq_len=inferred_params['max_seq_len']
    )
    
    # Create model
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
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("\n" + "=" * 80)
    print("HISTORY MODULE ANALYSIS")
    print("=" * 80)
    
    # Print learnable parameters
    print(f"\nLearnable History Parameters:")
    print(f"  recency_decay: {model.recency_decay.item():.4f}")
    print(f"  freq_weight: {model.freq_weight.item():.4f}")
    print(f"  history_scale: {model.history_scale.item():.4f}")
    print(f"  model_weight: {model.model_weight.item():.4f}")
    
    # Analyze predictions on test batches
    history_scores_list = []
    learned_scores_list = []
    final_scores_list = []
    correct_in_history = []
    correct_predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # Analyze first 10 batches
                break
            
            loc_seq = batch['loc_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            weekday_seq = batch['weekday_seq'].to(device)
            start_min_seq = batch['start_min_seq'].to(device)
            dur_seq = batch['dur_seq'].to(device)
            diff_seq = batch['diff_seq'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            # Get history scores
            history_scores = model.compute_history_scores(loc_seq, mask)
            
            # Get learned model predictions
            # We need to extract the learned component
            batch_size, seq_len = loc_seq.shape
            
            # Feature extraction (copied from model forward)
            import math
            import torch.nn.functional as F
            
            loc_emb = model.loc_emb(loc_seq)
            user_emb = model.user_emb(user_seq)
            
            hours = start_min_seq / 60.0
            time_rad = (hours / 24.0) * 2 * math.pi
            time_sin = torch.sin(time_rad)
            time_cos = torch.cos(time_rad)
            dur_norm = torch.log1p(dur_seq) / 8.0
            wd_rad = (weekday_seq.float() / 7.0) * 2 * math.pi
            wd_sin = torch.sin(wd_rad)
            wd_cos = torch.cos(wd_rad)
            diff_norm = diff_seq.float() / 7.0
            
            temporal_feats = torch.stack([time_sin, time_cos, dur_norm, wd_sin, wd_cos, diff_norm], dim=-1)
            temporal_emb = model.temporal_proj(temporal_feats)
            
            x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
            x = model.input_norm(x)
            x = x + model.pe[:seq_len, :].unsqueeze(0)
            x = model.dropout(x)
            
            attn_mask = ~mask
            attn_out, _ = model.attn(x, x, x, key_padding_mask=attn_mask)
            x = model.norm1(x + model.dropout(attn_out))
            ff_out = model.ff(x)
            x = model.norm2(x + model.dropout(ff_out))
            
            seq_lens = mask.sum(dim=1) - 1
            indices_gather = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, model.d_model)
            last_hidden = torch.gather(x, 1, indices_gather).squeeze(1)
            
            learned_logits = model.predictor(last_hidden)
            learned_logits_normalized = F.softmax(learned_logits, dim=1) * model.num_locations
            
            # Final predictions
            final_logits = history_scores + model.model_weight * learned_logits_normalized
            
            # Calculate statistics
            for j in range(batch_size):
                target = targets[j].item()
                
                # Get scores for target location
                hist_score = history_scores[j, target].item()
                learned_score = learned_logits_normalized[j, target].item()
                final_score = final_logits[j, target].item()
                
                history_scores_list.append(hist_score)
                learned_scores_list.append(learned_score)
                final_scores_list.append(final_score)
                
                # Check if target is in history
                seq = loc_seq[j].cpu().numpy()
                mask_j = mask[j].cpu().numpy()
                valid_locs = seq[mask_j]
                correct_in_history.append(int(target in valid_locs))
                
                # Check if prediction is correct
                pred = final_logits[j].argmax().item()
                correct_predictions.append(int(pred == target))
    
    print(f"\n{len(history_scores_list)} samples analyzed")
    print(f"\nScore Statistics:")
    print(f"  History scores - Mean: {np.mean(history_scores_list):.4f}, Std: {np.std(history_scores_list):.4f}")
    print(f"  Learned scores - Mean: {np.mean(learned_scores_list):.4f}, Std: {np.std(learned_scores_list):.4f}")
    print(f"  Final scores   - Mean: {np.mean(final_scores_list):.4f}, Std: {np.std(final_scores_list):.4f}")
    
    print(f"\nTarget Coverage:")
    print(f"  Targets in history: {np.mean(correct_in_history):.2%}")
    print(f"  Correct predictions: {np.mean(correct_predictions):.2%}")
    
    # Analyze contribution
    history_contribution = np.mean(history_scores_list) / (np.mean(history_scores_list) + np.mean(learned_scores_list))
    learned_contribution = np.mean(learned_scores_list) / (np.mean(history_scores_list) + np.mean(learned_scores_list))
    
    print(f"\nRelative Contribution:")
    print(f"  History module: {history_contribution:.2%}")
    print(f"  Learned module: {learned_contribution:.2%}")
    
    # Check if history scores are effectively zero
    hist_nonzero = np.sum(np.array(history_scores_list) > 0.001)
    print(f"\nNon-zero history scores: {hist_nonzero}/{len(history_scores_list)} ({hist_nonzero/len(history_scores_list):.2%})")
    
    print("=" * 80)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ANALYZING DIY DATASET")
    print("=" * 80)
    
    # Find best checkpoint for DIY
    diy_run_dir = Path("results/checkpoints")
    diy_checkpoints = list(diy_run_dir.glob("*diy*/best_model.pt"))
    
    if diy_checkpoints:
        # Use most recent
        diy_checkpoint = sorted(diy_checkpoints, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Using checkpoint: {diy_checkpoint}")
        analyze_history_contribution('configs/diy.yaml', str(diy_checkpoint))
    else:
        print("No DIY checkpoint found. Please train first.")
    
    print("\n" + "=" * 80)
    print("ANALYZING GEOLIFE DATASET")
    print("=" * 80)
    
    # Find best checkpoint for Geolife
    geo_checkpoints = list(diy_run_dir.glob("*geolife*/best_model.pt"))
    
    if geo_checkpoints:
        geo_checkpoint = sorted(geo_checkpoints, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Using checkpoint: {geo_checkpoint}")
        analyze_history_contribution('configs/geolife_default.yaml', str(geo_checkpoint))
    else:
        print("No Geolife checkpoint found. Please train first.")
