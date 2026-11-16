# Credit Card Fraud Detection — Hybrid Graph Deep Learning

**Portfolio research project** comparing graph-based and anomaly detection approaches to imbalanced credit-card fraud detection.

## Problem

Fraud accounts for only a small fraction of transactions. Models can show high ROC-AUC while delivering poor precision or F1 at the chosen alert threshold. We therefore report precision, recall, F1, ROC-AUC and PR-AUC separately.

## Model and evidence

- Primary checkpoint: original **Graph Attention + VAE + Bidirectional GRU** architecture from `HybridGNNVAEGRU_RAM_Fixed.ipynb`.
- Contains learned model weights, feature schema, train-only scaler metadata, validation-selected threshold and stored evaluation metrics.
- **Do not confuse this checkpoint with `best_model.pt` (weights only) or `best_ganomaly.pth` (different architecture).**
- Baseline experiments: OmniAnomaly, DAEGMM/GANomaly and Temporal Graph Network, discussed in [experiment report](docs/EXPERIMENT_RESULTS.md).

## Dataset

Public credit-card fraud dataset has **284,807 transactions, 492 fraud cases** (0.173%). Full training dataset is not bundled in this release. `data/creditcard_sample.csv` is an illustrative sampled subset (not a held-out test set).

## Setup

```bash
python -m venv .venv
# activate the venv for your OS
pip install -r requirements.txt
streamlit run dashboard.py
```

The interactive app initially displays stored validation/test metrics and then lets you run inference on a sample. **Inference requires PyTorch Geometric**, a compatible Python/Torch environment and the included `model/fraud_model.pt` checkpoint.

## Model details

- Feature creation: original V1–V28 and Amount; Time-derived `NormTime`, hour sine/cosine and three interaction features.
- Scalers fit on train only. Threshold selected using validation labels only.
- Graph is constructed within **128-row batches**; predictions are **batch-context dependent** and cannot be interpreted as independent, calibrated per-transaction fraud probabilities.
- The saved held-out test partition was only **3,187 rows including 15 fraud cases**. These results should not be extrapolated to the entire credit-card dataset.
- These are research/demo artifacts, not an operational financial decision system.

## Tests

```bash
pytest -q
```

Tests validate checkpoint format, feature preprocessing, schema, calculations, and saved metrics. Full neural-network inference needs the `torch-geometric` dependency.

## Reproducibility and limitations

See [experiment results](docs/EXPERIMENT_RESULTS.md) for metric interpretation and limitations. Stored checkpoint scores are **not independent replication**. To reproduce, run the fixed notebook with matching data, seed and dependency versions and check on untouched test data. The older hardcoded comparison charts do not constitute new run results.
