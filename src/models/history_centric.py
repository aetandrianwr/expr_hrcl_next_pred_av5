"""
History-Centric Next-Location Predictor

Core insight: 83.81% of next locations are already in the visit history.

Strategy:
1. Identify candidate locations from history
2. Score them using:
   - Recency (exponential decay)
   - Frequency in sequence
   - Learned transition patterns
   - Temporal context
3. Combine history scores with learned model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HistoryCentricModel(nn.Module):
    """
    Model that heavily prioritizes locations from visit history.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_locations = config.num_locations
        self.d_model = 80  # Compact
        
        # Core embeddings
        self.loc_emb = nn.Embedding(config.num_locations, 56, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, 12, padding_idx=0)
        
        # Compact temporal encoder
        self.temporal_proj = nn.Linear(6, 12)  # sin/cos time, dur, sin/cos wd, gap
        
        # Input fusion: 56 + 12 + 12 = 80
        self.input_norm = nn.LayerNorm(80)
        
        # Positional encoding
        pe = torch.zeros(60, 80)
        position = torch.arange(0, 60, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 80, 2).float() * (-math.log(10000.0) / 80))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Very compact transformer
        self.attn = nn.MultiheadAttention(80, 4, dropout=0.35, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(80, 160),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(160, 80)
        )
        self.norm1 = nn.LayerNorm(80)
        self.norm2 = nn.LayerNorm(80)
        self.dropout = nn.Dropout(0.35)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(80, 160),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(160, config.num_locations)
        )
        
        # History scoring parameters (learnable) - MAXIMUM history bias for 50%+
        self.recency_decay = nn.Parameter(torch.tensor(0.55))  # Maximum recency bias
        self.freq_weight = nn.Parameter(torch.tensor(3.0))     # Maximum frequency weight
        self.history_scale = nn.Parameter(torch.tensor(15.0))  # Maximum history boost
        self.model_weight = nn.Parameter(torch.tensor(0.15))   # Minimal learned weight
        
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
    
    def compute_history_scores(self, loc_seq, mask):
        """
        Compute history-based scores for all locations.
        
        Returns:
            history_scores: (batch_size, num_locations) - scores for each location
        """
        batch_size, seq_len = loc_seq.shape
        
        # Initialize score matrix
        recency_scores = torch.zeros(batch_size, self.num_locations, device=loc_seq.device)
        frequency_scores = torch.zeros(batch_size, self.num_locations, device=loc_seq.device)
        
        # Compute recency and frequency scores
        for t in range(seq_len):
            locs_t = loc_seq[:, t]  # (B,)
            valid_t = mask[:, t].float()  # (B,)
            
            # Recency: exponential decay from the end
            time_from_end = seq_len - t - 1
            recency_weight = torch.pow(self.recency_decay, time_from_end)
            
            # Update recency scores (max over time for each location)
            indices = locs_t.unsqueeze(1)  # (B, 1)
            values = (recency_weight * valid_t).unsqueeze(1)  # (B, 1)
            
            # For each location, keep the maximum recency (most recent visit)
            current_scores = torch.zeros(batch_size, self.num_locations, device=loc_seq.device)
            current_scores.scatter_(1, indices, values)
            recency_scores = torch.maximum(recency_scores, current_scores)
            
            # Update frequency scores (sum over time)
            frequency_scores.scatter_add_(1, indices, valid_t.unsqueeze(1))
        
        # Normalize frequency scores
        max_freq = frequency_scores.max(dim=1, keepdim=True)[0].clamp(min=1.0)
        frequency_scores = frequency_scores / max_freq
        
        # Combine recency and frequency
        history_scores = recency_scores + self.freq_weight * frequency_scores
        history_scores = self.history_scale * history_scores
        
        return history_scores
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        batch_size, seq_len = loc_seq.shape
        
        # === Compute history-based scores ===
        history_scores = self.compute_history_scores(loc_seq, mask)
        
        # === Learned model ===
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
        x = self.dropout(x)
        
        # Transformer layer
        attn_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1
        indices_gather = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, indices_gather).squeeze(1)
        
        # Learned logits
        learned_logits = self.predictor(last_hidden)
        
        # === Ensemble: History + Learned ===
        # Normalize learned logits to similar scale as history scores
        learned_logits_normalized = F.softmax(learned_logits, dim=1) * self.num_locations
        
        # Combine with learned weight
        final_logits = history_scores + self.model_weight * learned_logits_normalized
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
