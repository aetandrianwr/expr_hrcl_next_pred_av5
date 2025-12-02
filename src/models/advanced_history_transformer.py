"""
Advanced History-Aware Transformer for Large Vocabulary

Key innovations:
1. Improved history scoring with temporal decay and recency weighting
2. Frequency-aware prediction using top-K dense head + candidate scoring
3. Multi-scale temporal features
4. Optimized ensemble strategy
5. User-location co-occurrence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict


class AdvancedHistoryTransformer(nn.Module):
    """
    Advanced model combining:
    - Enhanced history module with temporal weighting
    - Frequency-aware dual prediction head
    - Rich temporal features
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.num_locations = config.num_locations
        self.num_users = config.num_users
        
        # Model dimensions - optimized for 3M budget
        self.d_model = 128
        loc_emb_dim = 56
        user_emb_dim = 16
        temporal_dim = self.d_model - loc_emb_dim - user_emb_dim
        
        # Embeddings
        self.loc_emb = nn.Embedding(config.num_locations, loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, user_emb_dim, padding_idx=0)
        
        # Multi-scale temporal encoding
        self.temporal_proj = nn.Sequential(
            nn.Linear(10, temporal_dim),  # Richer temporal features
            nn.LayerNorm(temporal_dim),
            nn.ReLU()
        )
        
        # Input normalization
        self.input_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(0.15)
        
        # Positional encoding
        max_len = 150  # Increased for DIY dataset
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer - 4 layers for depth
        nhead = 8
        num_layers = 4
        dim_feedforward = 256
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dual prediction head
        # Top-K frequent locations (covers ~80% of targets)
        self.top_k = 2500
        
        # Dense head for frequent locations
        self.dense_proj = nn.Sequential(
            nn.Linear(self.d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, self.top_k)
        )
        
        # Candidate scoring for rare locations
        self.candidate_proj = nn.Sequential(
            nn.Linear(self.d_model, loc_emb_dim),
            nn.LayerNorm(loc_emb_dim)
        )
        
        # History scoring parameters
        self.history_decay = nn.Parameter(torch.tensor(0.95))  # Learnable decay
        self.recency_weight = nn.Parameter(torch.tensor(2.0))  # Learnable recency boost
        
        # Ensemble weight (learnable)
        self.ensemble_alpha = nn.Parameter(torch.tensor(0.35))  # History weight
        
        # Frequency mapping (will be set during training)
        self.register_buffer('top_k_indices', torch.zeros(self.top_k, dtype=torch.long))
        self.register_buffer('location_frequencies', torch.ones(config.num_locations))
        
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
    
    def set_frequency_mapping(self, top_k_location_ids, location_frequencies):
        """Set the top-K frequent locations."""
        self.top_k_indices = torch.tensor(top_k_location_ids[:self.top_k], dtype=torch.long)
        self.location_frequencies = torch.tensor(location_frequencies, dtype=torch.float)
    
    def compute_enhanced_history_scores(self, loc_seq, mask):
        """
        Enhanced history scoring with:
        - Temporal decay (recent visits weighted more)
        - Frequency-based scoring
        - Position-based weighting
        """
        batch_size, seq_len = loc_seq.shape
        device = loc_seq.device
        
        # Initialize scores
        history_scores = torch.zeros(batch_size, self.num_locations, device=device)
        
        # Get sequence lengths
        seq_lens = mask.sum(dim=1)
        
        for i in range(batch_size):
            valid_len = seq_lens[i].item()
            if valid_len == 0:
                continue
            
            valid_locs = loc_seq[i, :valid_len]
            
            # Count occurrences with position weighting
            loc_counts = defaultdict(float)
            for pos, loc in enumerate(valid_locs.tolist()):
                if loc == 0:  # Skip padding
                    continue
                
                # Recency weight: more recent = higher weight
                recency = (pos + 1) / valid_len
                recency_boost = torch.sigmoid(self.recency_weight * (recency - 0.5)).item()
                
                # Temporal decay from last occurrence
                time_weight = self.history_decay ** (valid_len - pos - 1)
                
                # Combined weight
                weight = time_weight * (1.0 + recency_boost)
                loc_counts[loc] += weight
            
            # Convert to scores
            for loc, count in loc_counts.items():
                # Normalize by location frequency (boost rare locations)
                freq = self.location_frequencies[loc].item() if loc < len(self.location_frequencies) else 1.0
                freq_weight = 1.0 / (math.log(freq + 1) + 1.0)
                
                history_scores[i, loc] = count * freq_weight
        
        # Normalize scores
        max_scores = history_scores.max(dim=1, keepdim=True)[0]
        max_scores = torch.where(max_scores > 0, max_scores, torch.ones_like(max_scores))
        history_scores = history_scores / max_scores * 10.0  # Scale to ~10
        
        return history_scores
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        batch_size, seq_len = loc_seq.shape
        device = loc_seq.device
        
        # === Enhanced History Scores ===
        history_scores = self.compute_enhanced_history_scores(loc_seq, mask)
        
        # === Learned Representation ===
        # Embeddings
        loc_emb = self.loc_emb(loc_seq)
        user_emb = self.user_emb(user_seq)
        
        # Rich temporal features
        hours = start_min_seq / 60.0
        
        # Hour of day (cyclic)
        hour_rad = (hours / 24.0) * 2 * math.pi
        hour_sin = torch.sin(hour_rad)
        hour_cos = torch.cos(hour_rad)
        
        # Time of day category (morning/afternoon/evening/night)
        time_category = (hours / 6.0).long().clamp(0, 3)
        time_cat_onehot = F.one_hot(time_category, num_classes=4).float()
        
        # Weekday (cyclic)
        wd_rad = (weekday_seq.float() / 7.0) * 2 * math.pi
        wd_sin = torch.sin(wd_rad)
        wd_cos = torch.cos(wd_rad)
        
        # Duration (log-normalized)
        dur_norm = torch.log1p(dur_seq) / 8.0
        
        # Time difference (log-normalized)
        diff_norm = torch.log1p(diff_seq.float()) / 5.0
        
        # Combine temporal features (10 features total)
        temporal_feats = torch.cat([
            hour_sin.unsqueeze(-1),
            hour_cos.unsqueeze(-1),
            wd_sin.unsqueeze(-1),
            wd_cos.unsqueeze(-1),
            dur_norm.unsqueeze(-1),
            diff_norm.unsqueeze(-1),
            time_cat_onehot
        ], dim=-1)
        
        temporal_emb = self.temporal_proj(temporal_feats)
        
        # Combine all features
        x = torch.cat([loc_emb, user_emb, temporal_emb], dim=-1)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # Transformer
        attn_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1
        indices_gather = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, indices_gather).squeeze(1)
        
        # === Dual Prediction ===
        # 1. Dense prediction for top-K locations
        dense_logits = self.dense_proj(last_hidden)
        
        # 2. Candidate scoring for history locations
        query = self.candidate_proj(last_hidden)
        
        # Create full logits tensor
        learned_logits = torch.full((batch_size, self.num_locations), -20.0, device=device)
        
        # Scatter dense predictions
        if self.top_k_indices.device != device:
            self.top_k_indices = self.top_k_indices.to(device)
        
        learned_logits.scatter_(1, self.top_k_indices.unsqueeze(0).expand(batch_size, -1), dense_logits)
        
        # Add candidate scores for locations in history
        for i in range(batch_size):
            valid_locs = loc_seq[i][mask[i]].unique()
            if len(valid_locs) > 0:
                cand_embs = self.loc_emb(valid_locs)
                scores = torch.matmul(cand_embs, query[i])
                learned_logits[i, valid_locs] = torch.maximum(
                    learned_logits[i, valid_locs],
                    scores
                )
        
        # === Adaptive Ensemble ===
        # Use sigmoid to keep alpha in [0, 1]
        alpha = torch.sigmoid(self.ensemble_alpha)
        
        # Combine with adaptive weighting
        final_logits = alpha * history_scores + (1 - alpha) * learned_logits
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
