# Experiment results and validation status

## Source artifacts

- Uploaded `fraud_model.pt` (PyTorch checkpoint including weights and evaluation metadata).
- Uploaded `best_model.pt` (separate model state dict) and `best_ganomaly.pth` (GANomaly state dict).
- Uploaded `creditcard.csv` (284,807 rows; 492 fraud; 284,315 legitimate).
- Notebook `HybridGNNVAEGRU_RAM_Fixed.ipynb` (source for original hybrid architecture, preprocessing and batched anomaly scores).

## Checkpoint-reported results (not independently reproduced)

| Metric | Validation | Held-out Test |
|---|---:|---:|
| Rows | 3,187 | 3,187 |
| Fraud cases | 14 | 15 |
| Precision | 0.4062 | 0.3793 |
| Recall | 0.9286 | 0.7333 |
| F1 | 0.5652 | 0.5000 |
| ROC-AUC | 0.9942 | 0.7982 |
| PR-AUC (average precision) | 0.6879 | 0.5414 |
| Threshold | 0.269168 | 0.269168 |

The threshold was selected on validation data. The test F1 is **0.50**, **not >0.90**. No claim of >90% F1 should be made for this checkpoint.

### Derived held-out test confusion matrix

Assuming usual binary fraud-label semantics and rounding in the metrics: TP=11, FP=18, FN=4, TN=3,154. This matrix is **algebraically reconstructed from checkpoint-reported results**, not re-evaluated from raw predictions.

### Why ROC-AUC alone is insufficient

Because the data are extremely imbalanced, good class-ranking ability does not guarantee precision at a fixed threshold. Here precision is 0.3793 and recall is 0.7333. An alerting system with high false positives may be costly.

## Separate chart results (user-provided screenshots; different experiments)

The provided comparison plots display approximately: OmniAnomaly (F1 .20, AUC .85); DAEGMM (F1 .13, AUC .95); Temporal Graph Network (F1 .86, AUC .94); hybrid GNN-VAE-GRU (F1 .21, AUC .90). These values **differ from the uploaded hybrid checkpoint** and must not be mixed into a single benchmark without matching splits, implementations and metric definitions. A TGN trained model or per-row predictions were not supplied in this set of files, so the 0.86 TGN F1 cannot be independently checked here.

## Experimental risks

1. Small held-out positive sample: only 15 fraud cases; reported F1 has high sampling uncertainty.
2. Graph construction is within batches, so scores vary with neighbor context and batch ordering.
3. A bidirectional GRU applied to a one-token latent sequence does not establish true temporal sequence modeling.
4. This repository has no independently rerun full PyTorch Geometric model evaluation in the current runtime.
5. The dashboard sample is **not** the training notebook's holdout split; its statistics should not be used as test-set performance.
6. Full dataset can be redistributed only if licensing and provenance permit; therefore keep the full CSV outside git.

## Follow-up reproducibility steps

- Pin Python, PyTorch and torch-geometric versions from the working Colab environment.
- Save random seeds and explicit split indices, preprocessing schema, model config and checkpoint hashes.
- Run the hybrid on the untouched test split and export `y_true`, anomaly scores and predictions, preserving correct row indices.
- Select a threshold using validation data; never tune on test.
- Reproduce TGN evaluation under the exact same split before comparing it against hybrid.
- Add tests for order/batch sensitivity and consider a reference-graph approach for real deployment.

## Current verification

✅ Uploaded checkpoint loads safely with `torch.load(..., weights_only=True)`.
✅ Dataset row count and class distribution independently read.
✅ Checkpoint model metadata and preprocessing schema inspected.
⚠️ Model inference not executed end-to-end because `torch-geometric` was unavailable in the execution environment.
⚠️ GitHub repository has not been modified unless an authorized write succeeds separately.
