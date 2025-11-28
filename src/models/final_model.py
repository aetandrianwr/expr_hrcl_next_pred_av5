"""
Final Optimized Model for GeoLife Next-Location Prediction

Key strategy:
1. Ultra-compact Transformer (<500K params)
2. Explicit history matching (boost recently visited locations)
3. Strong dropout and regularization
4. Simple but effective architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FinalCompactModel(nn.Module):
    """
    Final compact model designed to maximize accuracy within <500K parameter budget.
    
    Architecture:
    - Minimal embeddings (location only, others are features)
    - 1-layer lightweight transformer
    - Explicit history attention
    - Strong regularization
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Core parameters
        self.num_locations = config.num_locations
        self.d_model = 96  # Reduced from config
        
        # Location embedding (main parameter consumer)
        # 1187 locations * 64 dims = 76K params
        self.loc_emb = nn.Embedding(config.num_locations, 64, padding_idx=0)
        
        # Compact feature encoders
        # User: just 16 dims
        self.user_emb = nn.Embedding(config.num_users, 16, padding_idx=0)
        
        # Temporal: cyclic sin/cos + learned projection (compact)
        self.temporal_proj = nn.Linear(6, 16)  # 6 features: time_sin, time_cos, dur, wd_sin, wd_cos, gap
        
        # Feature fusion: 64 + 16 + 16 = 96
        self.input_norm = nn.LayerNorm(96)
        
        # Positional encoding
        pe = torch.zeros(60, 96)
        position = torch.arange(0, 60, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 96, 2).float() * (-math.log(10000.0) / 96))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Single transformer layer (minimal)
        self.attn = nn.MultiheadAttention(96, 4, dropout=0.4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(96, 192),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(192, 96)
        )
        self.norm1 = nn.LayerNorm(96)
        self.norm2 = nn.LayerNorm(96)
        self.dropout = nn.Dropout(0.4)
        
        # Prediction head (compact)
        self.predictor = nn.Sequential(
            nn.Linear(96, 192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, config.num_locations)
        )
        
        # History boost parameters
        self.history_boost = nn.Parameter(torch.tensor(3.0))
        self.recency_decay = nn.Parameter(torch.tensor(0.85))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        batch_size, seq_len = loc_seq.shape
        
        # === Feature extraction ===
        loc_emb = self.loc_emb(loc_seq)  # (B, L, 64)
        user_emb = self.user_emb(user_seq)  # (B, L, 16)
        
        # Temporal features (compact cyclic encoding)
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
        temporal_emb = self.temporal_proj(temporal_feats)  # (B, L, 16)
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)  # (B, L, 96)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)
        
        # === Single transformer layer ===
        # Self-attention with masking
        attn_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        # === Get last valid position ===
        seq_lens = mask.sum(dim=1) - 1
        indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 96)
        last_hidden = torch.gather(x, 1, indices).squeeze(1)  # (B, 96)
        
        # === Base prediction ===
        logits = self.predictor(last_hidden)  # (B, num_locations)
        
        # === History-based boost ===
        # Create history score matrix
        history_scores = torch.zeros_like(logits)
        
        for t in range(seq_len):
            locs_t = loc_seq[:, t]
            valid_t = mask[:, t]
            
            # Recency weight (exponential decay)
            recency_weight = self.recency_decay ** (seq_len - t - 1)
            
            # Add boost
            boost_value = self.history_boost * recency_weight * valid_t.float()
            history_scores.scatter_add_(1, locs_t.unsqueeze(1), boost_value.unsqueeze(1))
        
        # Combine
        final_logits = logits + history_scores
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
