# GeoLife Next-Location Prediction

Hierarchical Transformer for next-location prediction on GeoLife dataset.

## Project Structure

```
.
├── src/
│   ├── configs/          # Configuration files
│   ├── data/            # Dataset and data loading
│   ├── models/          # Model architectures
│   ├── training/        # Training utilities
│   ├── evaluation/      # Metrics and evaluation
│   └── train.py         # Main training script
├── data/                # GeoLife dataset
├── trained_models/      # Saved model checkpoints
└── experiments/         # Experiment logs
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- scikit-learn

## Training

```bash
cd src
python train.py
```

## Model Architecture

- Hierarchical Transformer encoder with multi-modal embeddings
- Features: location, user, weekday, time-of-day, duration, time-gap
- Sinusoidal positional encoding
- Multi-head self-attention (8 heads, 4 layers)
- Parameter budget: < 500K
- No RNN/LSTM components

## Target

- Test Acc@1 ≥ 50%
