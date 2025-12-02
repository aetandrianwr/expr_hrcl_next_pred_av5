"""
Frequency-Aware Hybrid Model

Strategy:
1. Dense prediction head for top-K most frequent locations (covers ~80% of targets)
2. Candidate scoring for history locations not in top-K
3. Combines both seamlessly

This reduces parameters from 15K to ~2K while maintaining coverage!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyAwareModel(nn.Module):
    """
    Hybrid model: dense head for frequent locations, candidate scoring for rare ones.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_locations = config.num_locations
        self.top_k = getattr(config, 'top_k_locations', 2000)  # Top-K frequent locations
        
        # Get config parameters
        self.d_model = getattr(config, 'd_model', 128)
        loc_emb_dim = getattr(config, 'loc_emb_dim', 48)
        user_emb_dim = getattr(config, 'user_emb_dim', 16)
        nhead = getattr(config, 'nhead', 8)
        num_layers = getattr(config, 'num_layers', 3)
        dim_feedforward = getattr(config, 'dim_feedforward', 256)
        dropout = getattr(config, 'dropout', 0.2)
        
        # Calculate temporal dimension
        temporal_dim = self.d_model - loc_emb_dim - user_emb_dim
        
        # Core embeddings
        self.loc_emb = nn.Embedding(config.num_locations, loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, user_emb_dim, padding_idx=0)
        
        # Temporal encoder
        self.temporal_proj = nn.Linear(6, temporal_dim)
        
        # Input fusion
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Positional encoding
        max_len = getattr(config, 'max_seq_len', 100)
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dual prediction heads
        # 1. Dense head for top-K frequent locations
        self.dense_head = nn.Sequential(
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.top_k)
        )
        
        # 2. Candidate scorer for rare/history locations
        self.candidate_proj = nn.Linear(self.d_model, loc_emb_dim)
        
        # This will be set during training with actual frequency mapping
        self.register_buffer('top_k_indices', torch.zeros(self.top_k, dtype=torch.long))
        self.register_buffer('index_to_rank', torch.zeros(config.num_locations, dtype=torch.long))
        
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
    
    def set_frequency_mapping(self, top_k_location_ids):
        """
        Set the mapping from top-K indices to actual location IDs.
        Call this once after model creation with sorted location IDs by frequency.
        
        Args:
            top_k_location_ids: List/tensor of location IDs sorted by frequency (length=top_k)
        """
        self.top_k_indices = torch.tensor(top_k_location_ids, dtype=torch.long)
        
        # Create reverse mapping: location_id -> rank in top-K (-1 if not in top-K)
        index_to_rank = torch.full((self.num_locations,), -1, dtype=torch.long)
        for rank, loc_id in enumerate(top_k_location_ids):
            index_to_rank[loc_id] = rank
        self.index_to_rank = index_to_rank
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        batch_size, seq_len = loc_seq.shape
        device = loc_seq.device
        
        # === Learned representation ===
        # Feature extraction
        loc_emb = self.loc_emb(loc_seq)
        user_emb = self.user_emb(user_seq)
        
        # Temporal features
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
        temporal_emb = self.temporal_proj(temporal_feats)
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        attn_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1
        indices_gather = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, indices_gather).squeeze(1)  # (B, d_model)
        
        # === Dual prediction ===
        # 1. Dense prediction for top-K locations
        dense_logits = self.dense_head(last_hidden)  # (B, top_k)
        
        # 2. Candidate scoring for history locations
        query = self.candidate_proj(last_hidden)  # (B, loc_emb_dim)
        
        # Get unique history candidates for each sample
        history_scores = torch.full((batch_size, self.num_locations), float('-inf'), device=device)
        
        for i in range(batch_size):
            # Get unique locations from history
            valid_locs = loc_seq[i][mask[i]].unique()
            
            # Get embeddings and compute scores
            if len(valid_locs) > 0:
                cand_embs = self.loc_emb(valid_locs)  # (num_cands, loc_emb_dim)
                scores = torch.matmul(cand_embs, query[i])  # (num_cands,)
                history_scores[i, valid_locs] = scores
        
        # === Combine predictions ===
        # Create full logits tensor
        final_logits = torch.full((batch_size, self.num_locations), float('-inf'), device=device)
        
        # Scatter dense predictions to their actual location indices
        if self.top_k_indices.device != device:
            self.top_k_indices = self.top_k_indices.to(device)
            self.index_to_rank = self.index_to_rank.to(device)
        
        final_logits.scatter_(1, self.top_k_indices.unsqueeze(0).expand(batch_size, -1), dense_logits)
        
        # Add history scores (use torch.maximum to keep the best score)
        final_logits = torch.maximum(final_logits, history_scores)
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
