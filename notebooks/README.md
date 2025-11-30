# Project Notebooks

This directory contains comprehensive Jupyter notebooks documenting various aspects of the next-location prediction system.

## Available Notebooks

### 1. **complete_training_walkthrough.ipynb** ⭐
**Purpose**: Complete, self-contained walkthrough of HistoryCentricModel training from start to finish

**Features**:
- Fully executable without external project dependencies
- Exact replica of `train_model.py` logic and computations
- Step-by-step explanations of entire pipeline
- Covers: data loading → model architecture → training → evaluation
- Comprehensive metrics: Acc@k, MRR, NDCG, F1

**Contents**:
1. Data loading from pickle files
2. Dataset and DataLoader implementation
3. HistoryCentricModel architecture
4. Training loop with early stopping
5. Evaluation on test set
6. Results visualization

**Use Case**: Understanding and reproducing the complete training pipeline

---

### 2. **history_centric_model_walkthrough.ipynb**
**Purpose**: Deep dive into the HistoryCentricModel architecture

**Features**:
- Detailed model architecture explanation
- History-centric scoring mechanism
- Transformer components breakdown
- Parameter counting and efficiency analysis

---

### 3. **history_centric_model_input_pipeline.ipynb**
**Purpose**: Understanding input data flow through the model

**Features**:
- Input data structure and format
- Feature engineering and embeddings
- Attention mechanism visualization
- Forward pass walkthrough

---

### 4. **model_output_walkthrough.ipynb**
**Purpose**: Analyzing model predictions and outputs

**Features**:
- Output interpretation
- Prediction analysis
- History vs learned patterns comparison

---

### 5. **evaluation_pipeline_walkthrough.ipynb**
**Purpose**: Comprehensive evaluation metrics explanation

**Features**:
- Metric definitions and calculations
- Performance analysis
- Results interpretation

---

### 6. **geolife_preprocessing_complete_pipeline.ipynb**
**Purpose**: Data preprocessing pipeline documentation

**Features**:
- Raw data processing
- Feature extraction
- Dataset creation
- Train/val/test splitting

---

## Quick Start

To run the complete training walkthrough:

```bash
jupyter notebook complete_training_walkthrough.ipynb
```

Or in Google Colab:
1. Upload the notebook
2. Ensure data files are accessible at the specified paths
3. Run all cells sequentially

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- scikit-learn
- tqdm
- matplotlib (optional, for visualizations)

## Notes

- All notebooks use the GeoLife trajectory dataset
- Training notebook requires pickle data files in `data/geolife/`
- Notebooks are self-contained with minimal external dependencies
- Code matches production implementation in `src/` directory

## Data Files Required

For `complete_training_walkthrough.ipynb`:
- `data/geolife/geolife_transformer_7_train.pk`
- `data/geolife/geolife_transformer_7_validation.pk`
- `data/geolife/geolife_transformer_7_test.pk`

## Author

GeoLife Research Team

## Last Updated

November 30, 2025
