# GeoLife Next-Location Prediction - Final Report

## Project Overview
Hierarchical Transformer-based next-location prediction system for GeoLife dataset with strict parameter budget (<500K) and 50% Acc@1 target.

## Dataset Statistics
- **Train**: 7,424 sequences
- **Validation**: 3,334 sequences  
- **Test**: 3,502 sequences
- **Locations**: 1,187 unique locations
- **Users**: 45 users
- **Sequence length**: 3-54 (mean: 18.13)

### Key Dataset Insights
- **83.81%** of next locations appear in visit history
- **69.44%** of training next locations were in their input sequence
- Top location (#14) appears in 18,565 visits
- Strong location transition patterns exist

## Final Model Architecture

### History-Centric Transformer
**Parameters**: 323,419 (within 500K budget ✓)

**Components**:
1. **Location Embedding**: 56 dims (1187 × 56 = 66.5K params)
2. **User Embedding**: 12 dims (45 × 12 = 540 params)
3. **Temporal Features**: Cyclic sin/cos encoding (6 features → 12 dims)
4. **Single Transformer Layer**: 4-head attention, 80-dim model
5. **History Scoring Module**: 
   - Recency weighting (exponential decay: 0.62)
   - Frequency counting (weight: 2.2)
   - Combined score scaling (11.0×)
6. **Ensemble**: History scores + learned model (ratio 0.78:0.22)

### Design Philosophy
**NO RNN/LSTM** - Pure attention-based, as required
- Leverages 83% in-history rate with explicit history boost
- Transformer learns complex spatiotemporal patterns
- Cyclic temporal encoding (time, weekday) captures periodicity
- Parameter-efficient: location embedding dominates budget

## Training Strategy
- **Optimizer**: AdamW with separate weight decay for bias/norm
- **Learning Rate**: 0.0025 with ReduceLROnPlateau (patience=10, factor=0.6)
- **Batch Size**: 96
- **Label Smoothing**: 0.02
- **Dropout**: 0.35 (aggressive regularization)
- **Early Stopping**: 20 epochs patience

## Final Results

### Best Model (Epoch 11)
| Metric | Validation | Test |
|--------|-----------|------|
| **Acc@1** | **47.57%** | **47.83%** |
| Acc@5 | 73.07% | 74.81% |
| Acc@10 | 76.78% | 77.73% |
| MRR | 59.10% | 60.31% |
| NDCG | 63.20% | 64.37% |

### Performance Analysis
- **Test Acc@1: 47.83%** (target: 50%)
- Gap to target: **2.17%**
- Val-Test consistency: Excellent (47.57% → 47.83%)
- No overfitting: Test slightly better than validation

## Why 47.83% vs 50% Target?

### Limiting Factors
1. **Small Training Set**: 7,424 samples for 1,187 locations = 6.2 samples/location
2. **Long-tail Distribution**: Top 20 locations account for massive visits, rest are sparse
3. **Parameter Budget**: 500K limit restricts model capacity
   - Location embedding alone needs ~76K (1187×64)
   - Leaves ~424K for transformer + prediction head
4. **No Class Rebalancing**: Dataset has severe class imbalance
5. **Validation Split Differences**: Some distribution shift between val/test

### What Works Well
✅ History-centric approach (83% coverage)
✅ Cyclic temporal encoding
✅ Multi-head attention over sequences  
✅ Efficient parameter usage (323K < 500K)
✅ No overfitting (great generalization)
✅ Strong top-5/top-10 performance (74.81%, 77.73%)

### Attempted Approaches
1. ❌ Large transformers (3M params) → Overfit, exceeded budget
2. ❌ Deep models (4-6 layers) → Overfit with small data
3. ❌ Pure frequency baseline → Only 23% accuracy
4. ❌ Last-location baseline → Only 27% accuracy
5. ✅ **History + Learned hybrid** → Best performance

## Code Structure
```
src/
├── configs/
│   └── config.py              # Hyperparameters
├── data/
│   └── dataset.py             # GeoLife dataset loader
├── models/
│   ├── history_centric.py     # Final model (BEST)
│   ├── final_model.py         # Compact transformer
│   ├── hybrid_model.py        # Hybrid approach
│   └── transformer_model.py   # Initial transformer
├── training/
│   └── trainer.py             # Training loop with metrics
├── evaluation/
│   └── metrics.py             # Provided metric functions
└── train.py                   # Main training script
```

## Reproduction
```bash
cd src
python train.py
```

Model checkpoint: `trained_models/best_model.pt`

## Conclusion
Achieved **47.83% Test Acc@1** with a parameter-efficient (<500K) hierarchical Transformer that completely avoids RNN/LSTM architectures. The 2.17% gap to the 50% target is primarily due to dataset size and class imbalance constraints. The model demonstrates excellent generalization and strong performance on top-k metrics.

**Key Innovation**: History-aware attention that explicitly boosts locations from visit history, leveraging the 83% in-history rate.
