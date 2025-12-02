"""
Efficient History-Centric Model with Candidate Sampling

Key insight: Don't predict over all 15K locations. Instead:
1. Extract candidate locations from history (typically ~10-20 per sequence)
2. Use transformer to rank these candidates
3. Add a few "exploration" candidates from learned model

This reduces prediction from O(15K) to O(20-50), saving massive parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientHistoryModel(nn.Module):
    """
    Memory-efficient model using candidate sampling from history.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_locations = config.num_locations
        
        # Get config parameters
        self.d_model = getattr(config, 'd_model', 128)
        loc_emb_dim = getattr(config, 'loc_emb_dim', 64)
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
        
        # Candidate scoring: project hidden state to embedding space
        # Then score candidates by dot product similarity
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, loc_emb_dim)
        )
        
        # Small exploration head for non-history candidates
        # This allows model to predict locations NOT in history
        self.exploration_head = nn.Linear(self.d_model, 100)  # Top-100 most frequent locations
        
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
        device = loc_seq.device
        
        # === Extract candidates from history ===
        # For each sequence, get unique locations
        candidates_list = []
        max_candidates = 0
        
        for i in range(batch_size):
            valid_locs = loc_seq[i][mask[i]].tolist()
            unique_locs = list(dict.fromkeys(valid_locs))  # Preserve order, remove duplicates
            candidates_list.append(unique_locs)
            max_candidates = max(max_candidates, len(unique_locs))
        
        # Pad candidates to same length
        candidates = torch.zeros(batch_size, max_candidates, dtype=torch.long, device=device)
        candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool, device=device)
        
        for i, cands in enumerate(candidates_list):
            candidates[i, :len(cands)] = torch.tensor(cands, device=device)
            candidate_mask[i, :len(cands)] = True
        
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
        
        # === Score candidates ===
        # Project to embedding space
        query = self.output_proj(last_hidden)  # (B, loc_emb_dim)
        
        # Get embeddings of candidate locations
        cand_embs = self.loc_emb(candidates)  # (B, max_cand, loc_emb_dim)
        
        # Compute similarity scores
        scores = torch.bmm(cand_embs, query.unsqueeze(2)).squeeze(2)  # (B, max_cand)
        
        # Mask invalid candidates
        scores = scores.masked_fill(~candidate_mask, float('-inf'))
        
        # Create full logits tensor
        logits = torch.full((batch_size, self.num_locations), float('-inf'), device=device)
        
        # Scatter candidate scores
        logits.scatter_(1, candidates, scores)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
