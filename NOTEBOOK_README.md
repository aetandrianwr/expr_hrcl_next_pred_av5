# Model Output Walkthrough Notebook

## Overview

The `model_output_walkthrough.ipynb` notebook provides a comprehensive, step-by-step explanation of how the History-Centric Next-Location Prediction model processes input data and generates predictions.

## Key Features

### ✅ Completely Self-Contained
- **No external dependencies** on project scripts
- All model classes, functions, and utilities are defined directly in the notebook
- Can be run independently without importing from `src/` directory
- Perfect for learning, teaching, or sharing

### 📚 Comprehensive Coverage

The notebook walks through:

1. **Model Architecture** - Complete History-Centric model implementation
   - Embedding layers (locations, users)
   - Temporal feature encoding
   - Transformer architecture (self-attention + feedforward)
   - History scoring mechanism
   - Ensemble combination strategy

2. **Evaluation Metrics** - All metric functions included
   - Accuracy@K (K=1,3,5,10)
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG)
   - F1 Score

3. **Data Processing** - Dataset and DataLoader classes
   - Variable-length sequence handling
   - Padding and masking
   - Batch collation

4. **Step-by-Step Forward Pass**
   - Single sample walkthrough
   - History scoring visualization
   - Learned model processing
   - Ensemble combination
   - Final predictions

5. **Full Evaluation** - Complete test set analysis
   - Performance metrics
   - Prediction distribution visualization
   - History coverage analysis

## Structure

The notebook contains **28 cells**:
- **15 Markdown cells** with detailed explanations
- **13 Code cells** with complete, runnable implementations

## How to Use

### Option 1: View on GitHub
Simply open the notebook in the GitHub repository to read the documentation and code.

### Option 2: Run in Jupyter
```bash
cd notebooks/
jupyter notebook model_output_walkthrough.ipynb
```

### Option 3: Run in Google Colab
Upload the notebook to Google Colab for cloud execution with GPU support.

## What You'll Learn

### Core Concepts

1. **History-Based Prediction**
   - Why 83.81% of next locations are revisits
   - How recency and frequency scoring works
   - Exponential decay for temporal weighting

2. **Transformer Architecture**
   - Embedding layer design
   - Circular encoding for temporal features
   - Self-attention mechanism
   - Positional encoding

3. **Ensemble Strategy**
   - Why history dominates (~11x weight)
   - Role of learned model (~0.22x weight)
   - Normalization and combination

4. **Evaluation Metrics**
   - What each metric measures
   - How to interpret results
   - Trade-offs between metrics

## Code Organization

Each code cell is:
- **Self-contained**: Can run without dependencies
- **Well-documented**: Inline comments explain every step
- **Educational**: Designed for understanding, not just execution

Each markdown cell provides:
- **Context**: Why this step matters
- **Explanation**: What the code does
- **Insights**: Key takeaways and design decisions

## Example Outputs

The notebook produces:
- Model architecture summary
- Parameter counts and weight values
- Single sample analysis with detailed traces
- Top-K prediction comparisons
- Performance metrics across full test set
- Visualization of prediction distribution
- History coverage analysis

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Data Requirements

The notebook expects test data at:
```
/content/expr_hrcl_next_pred_av5/data/geolife/geolife_transformer_7_test.pk
```

And pre-trained weights at:
```
/content/expr_hrcl_next_pred_av5/trained_models/best_model.pt
```

**Note**: The notebook can run with randomly initialized weights to demonstrate the architecture even if pretrained weights are not available.

## Performance

On the GeoLife test set (3,502 samples), the trained model achieves:
- **Acc@1**: ~75-80% (exact match)
- **Acc@5**: ~90-95% (within top 5)
- **Acc@10**: ~95-98% (within top 10)
- **MRR**: ~80-85%
- **F1 Score**: ~75-80%

## Educational Value

This notebook is ideal for:
- **Students**: Learning about trajectory prediction and transformers
- **Researchers**: Understanding the model architecture
- **Practitioners**: Implementing similar systems
- **Reviewers**: Evaluating the approach

## Reproducibility

The notebook includes:
- Fixed random seeds for reproducibility
- Clear data paths and file requirements
- Explicit device configuration (CPU/GPU)
- Complete error handling

## Related Files

- **Training script**: `train_model.py` - Train the model from scratch
- **Model source**: `src/models/history_centric.py` - Production model code
- **Evaluation**: `src/evaluation/metrics.py` - Metric implementations
- **Config**: `configs/geolife_default.yaml` - Training configuration

## Citation

If you use this notebook in your research or teaching, please cite the original project.

---

**Created**: 2025-11-30  
**Version**: 1.0  
**Maintainer**: Project Team
