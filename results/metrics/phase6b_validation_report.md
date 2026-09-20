# Phase 6B — Validation-Based Configuration Selection Report
## 1. Purpose
Select **one locked configuration per model family** (SARIMA, LSTM, TCN) using the Phase 5 training period for fitting and the Phase 5 validation period for scoring. This is not a final model ranking across families.
## 2. Data used
- Target squares: `[5161, 5059, 5259]`
- Train: `2013-11-01 00:00` → `2013-12-08 23:50` (5472 rows/square)
- Validation: `2013-12-09 00:00` → `2013-12-15 23:50` (1008 rows/square)
- Sequence length: `144`
- Test period `2013-12-16 00:00` → `2013-12-22 23:50`: **held out**
## 3. Leakage controls
- Scalers fitted on training data only (Phase 5).
- Neural early stopping monitors validation loss only.
- SARIMA parameters estimated on the training series only.
- Selection metrics use validation predictions only.
- Test targets are never loaded into the selection logic.
- Normalization denominators use **training** mean traffic only.
## 4. Candidate configurations
### SARIMA
- **A**: order=(1, 0, 1), seasonal_order=(1, 0, 1, 144)
- **B**: order=(1, 0, 0), seasonal_order=(1, 0, 1, 144)
- **C**: order=(0, 0, 1), seasonal_order=(1, 0, 1, 144)
- **D**: order=(1, 0, 1), seasonal_order=(0, 0, 1, 144)

### LSTM
- **A**: units=32, layers=1, dropout=0.1, lr=0.001
- **B**: units=64, layers=1, dropout=0.1, lr=0.001
- **C**: units=32, layers=2, dropout=0.1, lr=0.001
- **D**: units=64, layers=1, dropout=0.2, lr=0.0005

### TCN
- **A**: filters=32, kernel=3, dilations=(1, 2, 4, 8), dropout=0.1, lr=0.001, RF=61 steps (610 min)
- **B**: filters=32, kernel=3, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1, lr=0.001, RF=253 steps (2530 min)
- **C**: filters=64, kernel=3, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1, lr=0.001, RF=253 steps (2530 min)
- **D**: filters=32, kernel=5, dilations=(1, 2, 4, 8, 16), dropout=0.1, lr=0.0005, RF=249 steps (2490 min)

## 5. Selection criterion
1. Primary: mean normalized validation MAE across squares  
   (`validation_MAE / training_mean_traffic`)  
2. Secondary: mean normalized validation RMSE  
3. Tie-breakers: lower mean raw MAE, lower total fit time, simpler configuration  
## 6. Validation results
### Selection summary
```
 model selected_candidate_id               selection_metric  n_squares_ok  mean_normalized_validation_mae  mean_normalized_validation_rmse  mean_raw_validation_mae  mean_raw_validation_rmse  mean_raw_validation_mape  total_fit_time_seconds                     note
SARIMA                  None mean_normalized_validation_mae             0                             NaN                              NaN                      NaN                       NaN                       NaN                     NaN No successful candidates
  LSTM                     B mean_normalized_validation_mae             3                        0.072465                         0.103828               100.999225                144.776642                  9.462800              259.957025                      NaN
   TCN                     A mean_normalized_validation_mae             3                        0.070453                         0.105150                98.151879                146.640447                  8.078593              349.072115                      NaN
```

### Selected candidate by square
```
model  square_id candidate_id status  validation_mae  validation_rmse  validation_mape  normalized_validation_mae  normalized_validation_rmse  fit_time_seconds  prediction_time_seconds
 LSTM       5161            B     ok      117.687272       171.319281        10.743886                   0.077868                    0.113354         64.079305                 0.510628
 LSTM       5059            B     ok       95.578627       129.673870         9.587359                   0.071203                    0.096603         86.035047                 0.686217
 LSTM       5259            B     ok       89.731776       133.336776         8.057155                   0.068324                    0.101526        109.842673                 1.036244
  TCN       5161            A     ok      114.584473       174.797762         9.448874                   0.075815                    0.115656        172.371098                 1.028620
  TCN       5059            A     ok       85.987133       126.317436         7.037494                   0.064058                    0.094103         91.965344                 0.983746
  TCN       5259            A     ok       93.884032       138.806145         7.749411                   0.071486                    0.105691         84.735673                 1.376205
```

## 7. Selected configuration for each model
### SARIMA
- Candidate: `None`
- Config: `{}`
### LSTM
- Candidate: `B`
- Config: `{"batch_size": 64, "dropout": 0.1, "early_stopping_patience": 5, "epochs": 20, "learning_rate": 0.001, "n_layers": 1, "seed": 42, "sequence_length": 144, "units": 64}`
### TCN
- Candidate: `A`
- Config: `{"batch_size": 64, "dilations": [1, 2, 4, 8], "dropout": 0.1, "early_stopping_patience": 5, "epochs": 20, "filters": 32, "kernel_size": 3, "learning_rate": 0.001, "receptive_field_minutes": 610, "receptive_field_steps": 61, "seed": 42, "sequence_length": 144}`

## 8. Naive baseline validation result
```
model candidate_id  square_id status  fit_time_seconds  prediction_time_seconds  is_research_model  validation_mae  validation_rmse  validation_mape  training_mean  normalized_validation_mae  normalized_validation_rmse  units  n_layers  dropout  learning_rate  batch_size  max_epochs  sequence_length  seed  actual_epochs  best_epoch  best_validation_loss  filters  kernel_size dilations  receptive_field_steps  receptive_field_minutes order seasonal_order method  maxiter error convergence_status fit_warning
Naive  persistence       5161     ok          0.002222                 0.005485              False      116.668542       175.156604         9.413001    1511.365465                   0.077194                    0.115893    NaN       NaN      NaN            NaN         NaN         NaN              NaN   NaN            NaN         NaN                   NaN      NaN          NaN       NaN                    NaN                      NaN   NaN            NaN    NaN      NaN   NaN                NaN         NaN
Naive  persistence       5059     ok          0.000945                 0.006707              False       99.987246       145.779672         8.356898    1342.332386                   0.074488                    0.108602    NaN       NaN      NaN            NaN         NaN         NaN              NaN   NaN            NaN         NaN                   NaN      NaN          NaN       NaN                    NaN                      NaN   NaN            NaN    NaN      NaN   NaN                NaN         NaN
Naive  persistence       5259     ok          0.002422                 0.006309              False       89.957916       132.199207         7.760029    1313.322054                   0.068496                    0.100660    NaN       NaN      NaN            NaN         NaN         NaN              NaN   NaN            NaN         NaN                   NaN      NaN          NaN       NaN                    NaN                      NaN   NaN            NaN    NaN      NaN   NaN                NaN         NaN
```

The naive baseline is a reference only and was not tuned.

## 9. TCN receptive-field analysis
Receptive field uses `1 + 2*(kernel_size-1)*sum(dilations)` because each residual block contains two causal dilated convolutions.

- Candidate A: RF = 61 steps (610 minutes); covers full L=144 window: False
- Candidate B: RF = 253 steps (2530 minutes); covers full L=144 window: True
- Candidate C: RF = 253 steps (2530 minutes); covers full L=144 window: True
- Candidate D: RF = 249 steps (2490 minutes); covers full L=144 window: True

A wider receptive field is **not** assumed to be better; selection is empirical.

## 10. Timing
```
model  count         sum       mean        max
 LSTM     12 2115.708504 176.309042 603.341956
  TCN     12 1691.973624 140.997802 342.720432
```

Failed candidate×square rows: **12**
```
 model candidate_id  square_id                                                                                                                                                                                       error
SARIMA            A       5161                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2448) and data type float64
SARIMA            A       5059                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
SARIMA            A       5259                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2448) and data type float64
SARIMA            B       5161                                                                                     MemoryError: Unable to allocate 393. MiB for an array with shape (145, 145, 2449) and data type float64
SARIMA            B       5059 RuntimeError: SARIMA fit failed for order=(1, 0, 0), seasonal_order=(1, 0, 1, 144): MemoryError: Unable to allocate 231. MiB for an array with shape (145, 145, 1441) and data type float64
SARIMA            B       5259                                                                                     MemoryError: Unable to allocate 393. MiB for an array with shape (145, 145, 2449) and data type float64
SARIMA            C       5161                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
SARIMA            C       5059                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
SARIMA            C       5259                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
SARIMA            D       5161                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
SARIMA            D       5059                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
SARIMA            D       5259                                                                                     MemoryError: Unable to allocate 398. MiB for an array with shape (146, 146, 2449) and data type float64
```

## 11. Reproducibility information
- Random seed: `42`
- Python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- TensorFlow: `2.20.0`
- statsmodels: `0.15.0`
- NumPy: `2.2.5`
- pandas: `2.3.0`
- Platform: `Windows-11-10.0.22631-SP0`
- GPU devices: `[]`

## 12. Statement on the final test period
**The Dec 16–22 test period was NOT used** for fitting, early stopping, normalization scales, hyperparameter selection, or model ranking in Phase 6B.

Phase 6B does **not** declare a winning model family. Cross-family comparison on the held-out test week belongs to a later phase.
