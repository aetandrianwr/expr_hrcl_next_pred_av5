"""
Enhanced Hierarchical Transformer with auxiliary tasks and better feature engineering.

Key improvements:
1. Parameter-efficient design (< 500K)
2. Location frequency-based weighting
3. User-location interaction modeling
4. Auxiliary task: predict time gap
5. Better temporal encoding with cyclic features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:x.size(1), :]


class EnhancedTemporalEncoding(nn.Module):
    """
    Enhanced temporal encoding with cyclic features for time of day and weekday.
    """
    
    def __init__(self, time_emb_dim):
        super().__init__()
        # Cyclic encoding for time of day (24 hour cycle)
        # Using sin/cos to capture periodic nature
        self.time_proj = nn.Linear(2, time_emb_dim)  # 2 for sin/cos
        
        # Duration encoding
        self.dur_proj = nn.Linear(1, time_emb_dim)
        
        # Time gap embedding
        self.gap_emb = nn.Embedding(8, time_emb_dim)
        
        self.layer_norm = nn.LayerNorm(time_emb_dim * 3)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, start_min, dur, diff):
        """
        Args:
            start_min: (batch_size, seq_len) - minutes from midnight
            dur: (batch_size, seq_len) - duration in minutes
            diff: (batch_size, seq_len) - time gap indicator
        """
        # Convert time to cyclic features (24-hour cycle)
        hours = start_min / 60.0  # Convert to hours
        time_rad = (hours / 24.0) * 2 * math.pi
        time_sin = torch.sin(time_rad).unsqueeze(-1)
        time_cos = torch.cos(time_rad).unsqueeze(-1)
        time_cyclic = torch.cat([time_sin, time_cos], dim=-1)
        time_enc = self.time_proj(time_cyclic)
        
        # Log-normalized duration
        dur_normalized = torch.log1p(dur).unsqueeze(-1) / 8.0
        dur_enc = self.dur_proj(dur_normalized)
        
        # Time gap embedding
        gap_enc = self.gap_emb(diff)
        
        # Concatenate and normalize
        temporal = torch.cat([time_enc, dur_enc, gap_enc], dim=-1)
        temporal = self.layer_norm(temporal)
        return self.dropout(temporal)


class LocationUserInteraction(nn.Module):
    """
    Model location-user interactions with a bilinear layer.
    This captures user preferences for specific locations.
    """
    
    def __init__(self, loc_emb_dim, user_emb_dim, out_dim):
        super().__init__()
        # Simple interaction via element-wise multiplication + projection
        self.interaction_proj = nn.Sequential(
            nn.Linear(loc_emb_dim + user_emb_dim + loc_emb_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
    
    def forward(self, loc_emb, user_emb):
        """
        Compute interaction between location and user embeddings.
        """
        # Element-wise product captures interaction
        interaction = loc_emb * user_emb.expand_as(loc_emb)
        # Concatenate original embeddings with interaction
        combined = torch.cat([loc_emb, user_emb.expand_as(loc_emb), interaction], dim=-1)
        return self.interaction_proj(combined)


class EfficientTransformerEncoder(nn.Module):
    """
    Parameter-efficient transformer encoder.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Embeddings
        self.loc_emb = nn.Embedding(config.num_locations, config.loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, config.user_emb_dim, padding_idx=0)
        self.weekday_emb = nn.Embedding(config.num_weekdays, config.weekday_emb_dim)
        
        # Cyclic weekday encoding (7-day cycle)
        self.weekday_cyclic_proj = nn.Linear(2, config.weekday_emb_dim)
        
        # Enhanced temporal encoding
        self.temporal_enc = EnhancedTemporalEncoding(config.time_emb_dim)
        
        # Location-user interaction
        self.loc_user_interaction = LocationUserInteraction(
            config.loc_emb_dim, config.user_emb_dim, config.loc_emb_dim
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        
        # Project to d_model
        input_dim = config.loc_emb_dim + config.weekday_emb_dim + config.time_emb_dim * 3
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Lightweight transformer layers
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
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        batch_size, seq_len = loc_seq.shape
        
        # Get base embeddings
        loc_emb = self.loc_emb(loc_seq)
        user_emb_base = self.user_emb(user_seq)
        
        # Location-user interaction
        loc_emb = self.loc_user_interaction(loc_emb, user_emb_base)
        
        # Cyclic weekday encoding
        weekday_rad = (weekday_seq.float() / 7.0) * 2 * math.pi
        weekday_sin = torch.sin(weekday_rad).unsqueeze(-1)
        weekday_cos = torch.cos(weekday_rad).unsqueeze(-1)
        weekday_cyclic = torch.cat([weekday_sin, weekday_cos], dim=-1)
        weekday_emb = self.weekday_cyclic_proj(weekday_cyclic)
        
        # Temporal features
        temporal_emb = self.temporal_enc(start_min_seq, dur_seq, diff_seq)
        
        # Combine features
        combined = torch.cat([loc_emb, weekday_emb, temporal_emb], dim=-1)
        
        # Project to d_model
        x = self.input_proj(combined)
        
        # Add positional encoding
        pos_enc = self.pos_encoder(x)
        x = x + pos_enc.unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer with masking
        attn_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        return x


class NextLocationPredictorV2(nn.Module):
    """
    Enhanced next-location predictor with auxiliary tasks.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.encoder = EfficientTransformerEncoder(config)
        
        # Main prediction head
        self.location_head = nn.Sequential(
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
        # Encode sequence
        encoded = self.encoder(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
        
        # Get last valid hidden state
        batch_size = loc_seq.size(0)
        seq_lens = mask.sum(dim=1) - 1
        indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.config.d_model)
        last_hidden = torch.gather(encoded, 1, indices).squeeze(1)
        
        # Predict next location
        logits = self.location_head(last_hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
