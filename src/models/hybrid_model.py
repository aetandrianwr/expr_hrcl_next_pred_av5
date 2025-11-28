"""
Hybrid History-Aware Transformer for next-location prediction.

Key innovations:
1. History attention: Boost scores for locations in sequence
2. Recency weighting: Recent locations get higher scores
3. Transition modeling: Learn location-to-location transitions
4. Frequency priors: Incorporate global location popularity
5. User preferences: User-specific location biases

Design: <500K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from collections import Counter


class HistoryAttentionModule(nn.Module):
    """
    Boost predictions for locations that appeared in the input sequence.
    Uses exponential decay for recency weighting.
    """
    
    def __init__(self, num_locations):
        super().__init__()
        # Learnable recency decay parameter
        self.decay = nn.Parameter(torch.tensor(0.9))
        self.boost_scale = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, logits, loc_seq, mask):
        """
        Args:
            logits: (B, num_locations)
            loc_seq: (B, L)
            mask: (B, L) - True for valid positions
        
        Returns:
            logits with history-based boost
        """
        batch_size, seq_len = loc_seq.shape
        num_locations = logits.size(1)
        
        # Create history score matrix
        history_scores = torch.zeros_like(logits)
        
        # For each position in sequence, add decayed boost
        for t in range(seq_len):
            # Get location at position t
            locs_t = loc_seq[:, t]  # (B,)
            valid_t = mask[:, t]  # (B,)
            
            # Compute recency weight (more recent = higher weight)
            recency_weight = self.decay ** (seq_len - t - 1)
            
            # Add boost to those locations
            history_scores.scatter_add_(
                1, 
                locs_t.unsqueeze(1), 
                (recency_weight * valid_t.float() * self.boost_scale).unsqueeze(1)
            )
        
        return logits + history_scores


class TransitionModel(nn.Module):
    """
    Model transitions from last location to next location.
    Uses a compact embedding approach to save parameters.
    """
    
    def __init__(self, num_locations, transition_dim=32):
        super().__init__()
        # Compact transition embeddings
        self.src_emb = nn.Embedding(num_locations, transition_dim)
        self.dst_proj = nn.Linear(transition_dim, num_locations)
        
    def forward(self, last_loc):
        """
        Args:
            last_loc: (B,) - last location in sequence
        
        Returns:
            transition_logits: (B, num_locations)
        """
        src_vec = self.src_emb(last_loc)  # (B, transition_dim)
        transition_logits = self.dst_proj(src_vec)  # (B, num_locations)
        return transition_logits


class CompactTransformerCore(nn.Module):
    """
    Compact transformer encoder core.
    """
    
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        """
        Args:
            x: (B, L, d_model)
            mask: (B, L)
        """
        # Add positional encoding
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        x = self.dropout(x)
        
        # Apply transformer
        attn_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        return x


class HybridLocationPredictor(nn.Module):
    """
    Hybrid model combining:
    - Transformer for sequence modeling
    - History attention for recency
    - Transition model for location-to-location patterns
    - User preferences
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config.d_model
        self.num_locations = config.num_locations
        
        # Embeddings - shared and compact
        self.loc_emb = nn.Embedding(config.num_locations, config.d_model // 2, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, config.d_model // 4, padding_idx=0)
        
        # Temporal features (compact cyclic encoding)
        self.temporal_dim = config.d_model // 4
        self.time_proj = nn.Linear(2, self.temporal_dim // 4)
        self.dur_proj = nn.Linear(1, self.temporal_dim // 4)
        self.gap_emb = nn.Embedding(8, self.temporal_dim // 4)
        self.weekday_proj = nn.Linear(2, self.temporal_dim // 4)
        
        # Feature fusion
        self.feature_fusion = nn.Linear(config.d_model, config.d_model)
        self.fusion_norm = nn.LayerNorm(config.d_model)
        
        # Transformer core
        self.transformer = CompactTransformerCore(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len
        )
        
        # Prediction heads
        self.sequence_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.LayerNorm(config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.num_locations)
        )
        
        # History attention module
        self.history_attention = HistoryAttentionModule(config.num_locations)
        
        # Transition model
        self.transition_model = TransitionModel(config.num_locations, transition_dim=32)
        
        # Mixing weights (learnable ensemble)
        self.mix_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3]))  # seq, history, transition
        
        self.config = config
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        """
        Forward pass with hybrid prediction.
        """
        batch_size, seq_len = loc_seq.shape
        
        # === Feature extraction ===
        loc_emb = self.loc_emb(loc_seq)
        user_emb = self.user_emb(user_seq)
        
        # Temporal features (cyclic)
        hours = start_min_seq / 60.0
        time_rad = (hours / 24.0) * 2 * math.pi
        time_feats = torch.stack([torch.sin(time_rad), torch.cos(time_rad)], dim=-1)
        time_enc = self.time_proj(time_feats)
        
        dur_norm = torch.log1p(dur_seq).unsqueeze(-1) / 8.0
        dur_enc = self.dur_proj(dur_norm)
        
        gap_enc = self.gap_emb(diff_seq)
        
        wd_rad = (weekday_seq.float() / 7.0) * 2 * math.pi
        wd_feats = torch.stack([torch.sin(wd_rad), torch.cos(wd_rad)], dim=-1)
        wd_enc = self.weekday_proj(wd_feats)
        
        temporal_emb = torch.cat([time_enc, dur_enc, gap_enc, wd_enc], dim=-1)
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
        x = self.feature_fusion(x)
        x = self.fusion_norm(x)
        
        # === Transformer encoding ===
        encoded = self.transformer(x, mask)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1
        indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(encoded, 1, indices).squeeze(1)
        
        # === Hybrid predictions ===
        # 1. Sequence-based prediction
        seq_logits = self.sequence_head(last_hidden)
        
        # 2. History-based boost
        history_logits = self.history_attention(seq_logits, loc_seq, mask)
        
        # 3. Transition-based prediction
        last_locs = torch.gather(loc_seq, 1, seq_lens.unsqueeze(1)).squeeze(1)
        transition_logits = self.transition_model(last_locs)
        
        # === Ensemble ===
        # Normalize mix weights
        weights = F.softmax(self.mix_weights, dim=0)
        
        final_logits = (
            weights[0] * seq_logits +
            weights[1] * history_logits +
            weights[2] * transition_logits
        )
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
