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
        
        # Get config parameters with defaults for backward compatibility
        self.d_model = getattr(config, 'd_model', 80)
        loc_emb_dim = getattr(config, 'loc_emb_dim', 56)
        user_emb_dim = getattr(config, 'user_emb_dim', 12)
        nhead = getattr(config, 'nhead', 4)
        num_layers = getattr(config, 'num_layers', 1)
        dim_feedforward = getattr(config, 'dim_feedforward', 160)
        dropout = getattr(config, 'dropout', 0.35)
        
        # Calculate temporal dimension to match d_model
        temporal_dim = self.d_model - loc_emb_dim - user_emb_dim
        
        # Core embeddings
        self.loc_emb = nn.Embedding(config.num_locations, loc_emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users, user_emb_dim, padding_idx=0)
        
        # Temporal encoder
        self.temporal_proj = nn.Linear(6, temporal_dim)  # sin/cos time, dur, sin/cos wd, gap
        
        # Input fusion: loc_emb_dim + user_emb_dim + temporal_dim = d_model
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Positional encoding
        max_len = getattr(config, 'max_seq_len', 100)
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer - support multiple layers
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
        self.num_layers = num_layers
        
        # Prediction head - lightweight but effective
        # Use a single linear layer to save parameters
        self.output_norm = nn.LayerNorm(self.d_model)
        self.predictor = nn.Linear(self.d_model, config.num_locations)
        
        
        # History scoring parameters (learnable) - dataset-specific initialization
        dataset_name = getattr(config, 'dataset_name', None)
        if dataset_name == 'diy_skip_first_part':
            # Tuned parameters for diy_skip_first_part dataset
            self.recency_decay = nn.Parameter(torch.tensor(0.75))
            self.freq_weight = nn.Parameter(torch.tensor(3.5))
            self.history_scale = nn.Parameter(torch.tensor(15.0))
            self.model_weight = nn.Parameter(torch.tensor(0.15))
        elif dataset_name == 'diy':
            # DIY dataset needs MORE learned model, LESS history
            # History can only achieve ~35% Acc@1, so learned model must dominate
            self.recency_decay = nn.Parameter(torch.tensor(0.70))
            self.freq_weight = nn.Parameter(torch.tensor(1.8))
            self.history_scale = nn.Parameter(torch.tensor(4.0))  # Reduce history influence
            self.model_weight = nn.Parameter(torch.tensor(3.0))   # Greatly increase learned model
        else:
            # Default parameters for geolife and other datasets
            self.recency_decay = nn.Parameter(torch.tensor(0.62))
            self.freq_weight = nn.Parameter(torch.tensor(2.2))
            self.history_scale = nn.Parameter(torch.tensor(11.0))
            self.model_weight = nn.Parameter(torch.tensor(0.22))
        
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
        
        # Transformer encoding
        attn_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1) - 1
        indices_gather = seq_lens.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
        last_hidden = torch.gather(x, 1, indices_gather).squeeze(1)
        
        # Learned logits
        last_hidden = self.output_norm(last_hidden)
        learned_logits = self.predictor(last_hidden)
        
        # === Ensemble: History + Learned ===
        # Normalize learned logits to similar scale as history scores
        learned_logits_normalized = F.softmax(learned_logits, dim=1) * self.num_locations
        
        # Combine with learned weight
        final_logits = history_scores + self.model_weight * learned_logits_normalized
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
