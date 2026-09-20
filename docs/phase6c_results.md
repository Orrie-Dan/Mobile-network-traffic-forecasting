# Phase 6C — Final Test Evaluation Report

- Source: Colab notebook `phase6c_final_evaluation.ipynb` (Drive outputs under `milan_traffic/results/phase6c/`)
- Test week: 2013-12-16 → 2013-12-22 (1008 bins/square)
- Best research model (mean nMAE): **SARIMA-A** (0.0528)

## Aggregate test metrics

| Model | mean MAE | mean RMSE | mean MAPE | mean nMAE | mean nRMSE |
|-------|----------|-----------|-----------|-----------|------------|
| SARIMA-A | 73.3 | 105.7 | 7.98% | 0.0528 | 0.0759 |
| LSTM-B | 91.3 | 126.0 | 12.16% | 0.0654 | 0.0905 |
| TCN-D | 181.7 | 264.7 | 17.80% | 0.1279 | 0.1866 |
| Naive | 83.3 | 119.7 | 8.47% | 0.0601 | 0.0860 |

## Per-square test metrics

| Model | Square | MAE | RMSE | MAPE | nMAE | nRMSE |
|-------|--------|-----|------|------|------|-------|
| LSTM-B | 5059 | 89 | 119 | 11.46% | 0.0663 | 0.0890 |
| LSTM-B | 5161 | 105 | 148 | 15.07% | 0.0692 | 0.0978 |
| LSTM-B | 5259 | 80 | 111 | 9.95% | 0.0607 | 0.0846 |
| Naive | 5059 | 81 | 114 | 7.96% | 0.0607 | 0.0852 |
| Naive | 5161 | 93 | 135 | 9.29% | 0.0616 | 0.0893 |
| Naive | 5259 | 76 | 110 | 8.15% | 0.0579 | 0.0835 |
| SARIMA-A | 5059 | 69 | 98 | 7.34% | 0.0515 | 0.0731 |
| SARIMA-A | 5161 | 81 | 123 | 8.00% | 0.0538 | 0.0815 |
| SARIMA-A | 5259 | 70 | 96 | 8.61% | 0.0532 | 0.0732 |
| TCN-D | 5059 | 142 | 211 | 11.98% | 0.1056 | 0.1568 |
| TCN-D | 5161 | 289 | 413 | 31.45% | 0.1910 | 0.2731 |
| TCN-D | 5259 | 114 | 170 | 9.97% | 0.0871 | 0.1298 |

## Protocol note

Configs were locked in Phase 6B on validation only. This phase refits locked
configs and scores Dec 16–22 once. Ranking is post-hoc reporting, not selection.
