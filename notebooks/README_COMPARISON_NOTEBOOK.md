# Comprehensive Model Comparison Notebook

## 📓 File
`comprehensive_model_comparison.ipynb`

## 🎯 Purpose
This notebook provides a **comprehensive, fair, and reproducible comparison** of the HistoryCentricModel against 6 baseline models for next-location prediction on the GeoLife dataset.

## ✨ Key Features

### Self-Contained Implementation
- ✅ **No external dependencies** on project scripts
- ✅ **All models implemented from scratch** within the notebook
- ✅ **Complete workflow** from data loading to results visualization
- ✅ **Fully reproducible** with fixed random seeds

### Fair Comparison Methodology
- ✅ **Same dataset**: Identical train/validation/test splits
- ✅ **Same features**: All models use the same input features
- ✅ **Similar capacity**: Neural models have comparable parameter counts (~100K-200K)
- ✅ **Same training**: Identical batch size, learning rate, early stopping
- ✅ **Same metrics**: All models evaluated on Acc@k, MRR, NDCG, F1

## 🤖 Models Compared

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | **HistoryCentricModel** | Hybrid | Combines history-based scoring with learned patterns |
| 2 | **Transformer-Only** | Neural | Pure transformer architecture without history priors |
| 3 | **LSTM** | RNN | Long Short-Term Memory recurrent network |
| 4 | **GRU** | RNN | Gated Recurrent Unit network |
| 5 | **RNN** | RNN | Simple recurrent neural network |
| 6 | **Markov Chain** | Statistical | First-order Markov model with transition probabilities |
| 7 | **Frequency Baseline** | Statistical | Predicts based on visit frequency in history |

## 📊 Evaluation Metrics

All models are evaluated on:
- **Accuracy@1, @5, @10**: Top-k prediction accuracy
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
- **F1 Score**: Weighted F1 for top-1 predictions

## 🚀 Quick Start

### 1. Open the Notebook
```bash
jupyter notebook comprehensive_model_comparison.ipynb
```

### 2. Update Data Paths
In the Configuration cell (Section 2), update:
```python
data_dir = '../data/geolife'  # Update to your data directory
```

### 3. Run All Cells
Execute cells sequentially from top to bottom. The notebook will:
1. Load and prepare the GeoLife dataset
2. Implement all 7 models from scratch
3. Train each model with identical settings
4. Evaluate on test set
5. Generate comparison visualizations

### 4. Review Results
Results are presented in:
- Comparison tables (Section 8)
- Bar charts showing all metrics
- Radar plots for top models
- Discussion and key findings (Section 9)

## ⏱️ Expected Runtime

| Scenario | Time | Notes |
|----------|------|-------|
| **Full training** | 2-4 hours | All models, 50 epochs each (on GPU) |
| **Quick test** | ~30 minutes | Reduce `num_epochs` to 10 in config |
| **Baselines only** | < 1 minute | Markov and Frequency (no training) |

## 📁 Notebook Structure

```
1. Setup and Imports                    # Libraries and configuration
2. Configuration                        # Unified config for all models
3. Dataset and DataLoader              # GeoLife data loading
4. Evaluation Metrics                  # Acc@k, MRR, NDCG, F1
5. Model Implementations               # All 7 models from scratch
   5.1 HistoryCentricModel            
   5.2 Transformer-Only               
   5.3 LSTM                           
   5.4 GRU                            
   5.5 Simple RNN                     
   5.6 Markov Chain                   
   5.7 Frequency Baseline             
6. Training Infrastructure             # Unified training loop
7. Model Training and Comparison       # Train all models
8. Results Analysis and Visualization  # Tables, charts, radar plots
9. Discussion and Key Findings         # Insights and analysis
10. Conclusion                         # Recommendations and next steps
```

**Total**: 49 cells (25 markdown + 24 code)

## 🛠️ Requirements

### Python Packages
```
torch >= 1.9.0
numpy >= 1.19.0
pandas >= 1.2.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
```

### Hardware
- **GPU recommended** for faster training
- **CPU compatible** but slower
- **~4GB RAM** minimum
- **~500MB disk** for data

## 📝 Customization

### Adjust Training Duration
In Section 2 (Configuration):
```python
num_epochs = 50  # Change to 10 for quick test, 120 for full training
```

### Modify Model Capacity
In Section 2 (Configuration):
```python
loc_emb_dim = 64      # Location embedding dimension
hidden_dim = 128      # RNN/LSTM/GRU hidden dimension
d_model = 128         # Transformer dimension
```

### Change Batch Size
```python
batch_size = 96       # Adjust based on GPU memory
```

## 📈 Example Results

The notebook generates:

1. **Comparison Table**
   ```
   Model                 Acc@1    Acc@5    Acc@10   MRR     NDCG    F1
   HistoryCentricModel   XX.XX%   XX.XX%   XX.XX%   XX.XX%  XX.XX%  XX.XX%
   Transformer-Only      XX.XX%   XX.XX%   XX.XX%   XX.XX%  XX.XX%  XX.XX%
   LSTM                  XX.XX%   XX.XX%   XX.XX%   XX.XX%  XX.XX%  XX.XX%
   ...
   ```

2. **Bar Charts**: 6 metrics × 7 models with best model highlighted

3. **Radar Plot**: Multi-metric comparison of top 4 models

## 🔍 Key Insights

The notebook demonstrates:
1. **History matters**: Explicit use of visit history improves performance
2. **Recency + Frequency**: Both signals are important
3. **Model capacity**: Similar parameter counts ensure fair comparison
4. **Attention mechanisms**: Help capture long-range dependencies
5. **Hybrid approach**: Combining priors with learning is effective

## 🔬 Research Use

This notebook is suitable for:
- **PhD research**: Comprehensive baseline comparisons
- **Paper submissions**: Fair evaluation methodology
- **Teaching**: Educational resource on location prediction
- **Prototyping**: Quick implementation of multiple approaches

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in configuration
- Reduce `num_epochs` for quicker training
- Use CPU if GPU memory is insufficient

### Import Errors
- Install missing packages: `pip install torch numpy pandas matplotlib scikit-learn`
- Check Python version: >= 3.7 recommended

### Data Loading Errors
- Verify data paths in configuration
- Ensure GeoLife pickle files exist
- Check file permissions

## 📚 References

- **GeoLife Dataset**: Microsoft Research Asia
- **Transformer**: Vaswani et al., "Attention Is All You Need" (2017)
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **GRU**: Cho et al. (2014)

## 📄 License

This notebook is part of the expr_hrcl_next_pred_av5 project.

## 🤝 Contributing

To extend this notebook:
1. Add new model implementations in Section 5
2. Update training loop in Section 7
3. Add new visualizations in Section 8
4. Update discussion in Section 9

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Created**: November 30, 2024  
**Status**: ✅ Complete and tested  
**Version**: 1.0  
**Cells**: 49 (25 markdown + 24 code)  
**Size**: ~71 KB
