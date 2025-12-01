"""
Compact Hierarchical Transformer - optimized for <500K parameters.

Design principles:
1. Share embeddings where possible
2. Minimal but effective feature fusion
3. Lightweight attention mechanism
4. Strong regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CompactPositionalEncoding(nn.Module):
    """Compact sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len):
        return self.pe[:seq_len, :]


class CompactTemporalFeatures(nn.Module):
    """
    Compact temporal feature encoding with cyclic time representation.
    """
    
    def __init__(self, d_model):
        super().__init__()
        # Time of day: map to cyclic features (sin/cos of hour)
        # Duration: log-normalized
        # Time gap: embedded
        # Weekday: cyclic (sin/cos)
        
        self.time_proj = nn.Linear(2, d_model // 4)  # sin/cos for time of day
        self.dur_proj = nn.Linear(1, d_model // 4)   # log duration
        self.gap_emb = nn.Embedding(8, d_model // 4)  # time gap
        self.weekday_proj = nn.Linear(2, d_model // 4)  # sin/cos for weekday
        
    def forward(self, start_min, dur, diff, weekday):
        """
        Returns: (batch_size, seq_len, d_model)
        """
        # Time of day cyclic encoding
        hours = start_min / 60.0
        time_rad = (hours / 24.0) * 2 * math.pi
        time_feats = torch.stack([torch.sin(time_rad), torch.cos(time_rad)], dim=-1)
        time_enc = self.time_proj(time_feats)
        
        # Duration (log-normalized)
        dur_norm = torch.log1p(dur).unsqueeze(-1) / 8.0
        dur_enc = self.dur_proj(dur_norm)
        
        # Time gap
        gap_enc = self.gap_emb(diff)
        
        # Weekday cyclic
        wd_rad = (weekday.float() / 7.0) * 2 * math.pi
        wd_feats = torch.stack([torch.sin(wd_rad), torch.cos(wd_rad)], dim=-1)
        wd_enc = self.weekday_proj(wd_feats)
        
        # Concatenate all temporal features
        return torch.cat([time_enc, dur_enc, gap_enc, wd_enc], dim=-1)


class CompactTransformer(nn.Module):
    """
    Extremely compact transformer for next-location prediction.
    Target: <500K parameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config.d_model
        
        # Location embedding (largest vocabulary)
        self.loc_emb = nn.Embedding(config.num_locations, config.d_model // 2, padding_idx=0)
        
        # User embedding (shared across sequence)
        self.user_emb = nn.Embedding(config.num_users, config.d_model // 4, padding_idx=0)
        
        # Temporal features (compact)
        self.temporal_features = CompactTemporalFeatures(config.d_model // 4)
        
        # Project combined features to d_model
        # loc (d_model//2) + user (d_model//4) + temporal (d_model//4) = d_model
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        self.pos_enc = CompactPositionalEncoding(config.d_model, config.max_seq_len)
        
        # Lightweight transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.LayerNorm(config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.num_locations)
        )
        
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
        Args:
            loc_seq: (B, L)
            user_seq: (B, L)
            weekday_seq: (B, L)
            start_min_seq: (B, L)
            dur_seq: (B, L)
            diff_seq: (B, L)
            mask: (B, L)
        """
        batch_size, seq_len = loc_seq.shape
        
        # Get embeddings
        loc_emb = self.loc_emb(loc_seq)  # (B, L, d_model//2)
        user_emb = self.user_emb(user_seq)  # (B, L, d_model//4)
        
        # Get temporal features
        temporal_emb = self.temporal_features(start_min_seq, dur_seq, diff_seq, weekday_seq)  # (B, L, d_model//4)
        
        # Concatenate all features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)  # (B, L, d_model)
        
        # Feature fusion
        x = self.feature_fusion(x)
        
        # Add positional encoding
        pos = self.pos_enc(seq_len).unsqueeze(0)  # (1, L, d_model)
        x = x + pos
        
        # Apply transformer with masking
        attn_mask = ~mask  # Invert for transformer (True = masked)
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # (B, L, d_model)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1  # Last valid index
        indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, indices).squeeze(1)  # (B, d_model)
        
        # Predict next location
        logits = self.predictor(last_hidden)  # (B, num_locations)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
