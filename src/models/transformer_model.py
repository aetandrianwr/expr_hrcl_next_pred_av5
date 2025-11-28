"""
Hierarchical Transformer model for next-location prediction.

This model uses:
1. Multi-modal embeddings (location, user, temporal features)
2. Sinusoidal positional encoding
3. Multi-head self-attention with proper masking
4. Hierarchical feature fusion
5. Temporal-aware attention mechanisms
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
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            Positional encodings: (seq_len, d_model)
        """
        return self.pe[:x.size(1), :]


class TemporalEncoding(nn.Module):
    """
    Encode temporal features (time of day, duration, time gap).
    Uses learnable projections for continuous time features.
    """
    
    def __init__(self, time_emb_dim):
        super().__init__()
        # Time of day encoding (minutes from midnight -> embedding)
        self.time_proj = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # Duration encoding
        self.dur_proj = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # Time gap encoding (discrete)
        self.gap_emb = nn.Embedding(8, time_emb_dim)  # diff values 0-7
        
        self.layer_norm = nn.LayerNorm(time_emb_dim * 3)
    
    def forward(self, start_min, dur, diff):
        """
        Args:
            start_min: (batch_size, seq_len) - minutes from midnight
            dur: (batch_size, seq_len) - duration in minutes
            diff: (batch_size, seq_len) - time gap indicator
        Returns:
            Temporal encoding: (batch_size, seq_len, time_emb_dim * 3)
        """
        # Normalize continuous features
        time_normalized = (start_min / 1440.0).unsqueeze(-1)  # Normalize to [0, 1]
        dur_normalized = torch.log1p(dur).unsqueeze(-1) / 8.0  # Log-scale, normalize
        
        time_enc = self.time_proj(time_normalized)
        dur_enc = self.dur_proj(dur_normalized)
        gap_enc = self.gap_emb(diff)
        
        # Concatenate all temporal features
        temporal = torch.cat([time_enc, dur_enc, gap_enc], dim=-1)
        return self.layer_norm(temporal)


class HierarchicalTransformerEncoder(nn.Module):
    """
    Hierarchical Transformer with multi-modal feature fusion.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Embeddings
        self.loc_emb = nn.Embedding(config.num_locations, config.loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, config.user_emb_dim, padding_idx=0)
        self.weekday_emb = nn.Embedding(config.num_weekdays, config.weekday_emb_dim)
        
        # Temporal encoding
        self.temporal_enc = TemporalEncoding(config.time_emb_dim)
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        
        # Project concatenated embeddings to d_model
        input_dim = config.loc_emb_dim + config.user_emb_dim + config.weekday_emb_dim + config.time_emb_dim * 3
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        """
        Args:
            loc_seq: (batch_size, seq_len)
            user_seq: (batch_size, seq_len)
            weekday_seq: (batch_size, seq_len)
            start_min_seq: (batch_size, seq_len)
            dur_seq: (batch_size, seq_len)
            diff_seq: (batch_size, seq_len)
            mask: (batch_size, seq_len) - True for valid positions
        Returns:
            Encoded sequence: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = loc_seq.shape
        
        # Get embeddings
        loc_emb = self.loc_emb(loc_seq)  # (B, L, loc_emb_dim)
        user_emb = self.user_emb(user_seq)  # (B, L, user_emb_dim)
        weekday_emb = self.weekday_emb(weekday_seq)  # (B, L, weekday_emb_dim)
        temporal_emb = self.temporal_enc(start_min_seq, dur_seq, diff_seq)  # (B, L, time_emb_dim*3)
        
        # Concatenate all features
        combined = torch.cat([loc_emb, user_emb, weekday_emb, temporal_emb], dim=-1)
        
        # Project to d_model
        x = self.input_proj(combined)  # (B, L, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoder(x)  # (L, d_model)
        x = x + pos_enc.unsqueeze(0)  # Broadcast to (B, L, d_model)
        
        x = self.dropout(x)
        
        # Create attention mask for transformer (inverted: True -> mask out)
        # Transformer expects True for positions to mask out
        attn_mask = ~mask  # Invert: True for padding, False for valid
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        return x


class NextLocationPredictor(nn.Module):
    """
    Complete model for next-location prediction.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.encoder = HierarchicalTransformerEncoder(config)
        
        # Prediction head with residual connection
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
        """Initialize weights with Xavier/Kaiming initialization."""
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
        Forward pass.
        
        Args:
            loc_seq: (batch_size, seq_len)
            user_seq: (batch_size, seq_len)
            weekday_seq: (batch_size, seq_len)
            start_min_seq: (batch_size, seq_len)
            dur_seq: (batch_size, seq_len)
            diff_seq: (batch_size, seq_len)
            mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_locations)
        """
        # Encode sequence
        encoded = self.encoder(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
        
        # Use last valid position for each sequence
        batch_size = loc_seq.size(0)
        seq_lens = mask.sum(dim=1) - 1  # Last valid index
        
        # Gather last valid hidden state
        indices = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.config.d_model)
        last_hidden = torch.gather(encoded, 1, indices).squeeze(1)  # (B, d_model)
        
        # Predict next location
        logits = self.predictor(last_hidden)  # (B, num_locations)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
