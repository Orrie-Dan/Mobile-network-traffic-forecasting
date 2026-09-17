# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Research Question

How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

## Project Overview

This university research project investigates sequential forecasting approaches for mobile network traffic using the Milan telecommunications dataset. The work focuses on understanding how different sequential models perform for one-step-ahead prediction, and whether performance differs across geographical areas with distinct traffic characteristics.

The project focuses on:

- Efficient handling of a large dataset
- Exploratory analysis of geographical and temporal traffic patterns
- Literature-informed model selection
- One-step-ahead forecasting
- Comparison across geographical areas
- Evaluation using MAE, RMSE, MAPE, and execution/training time
- Analysis of differences in model performance

## Dataset

This project uses the Milan telecommunications dataset from Harvard Dataverse.

The complete dataset contains 62 files and is approximately 19.4 GB. Because of its size, the raw dataset is **not** stored in this repository. Raw files are kept externally (Google Drive) and accessed for processing through Google Colab.

See [`data/raw/README.md`](data/raw/README.md) for details on external storage and access.

## Repository Structure

```
mobile-network-traffic-forecasting/
├── data/           # Raw (external), processed, and sample data directories
├── notebooks/      # Main experimental Jupyter notebook
├── src/            # Reusable Python modules for data, analysis, and evaluation
├── results/        # Figures, metrics, and prediction outputs
├── docs/           # Literature review and methodology notes
└── report/         # Formative report and related documents
```

| Path | Purpose |
|------|---------|
| `data/raw/` | Placeholder for raw dataset access notes (data stored externally) |
| `data/processed/` | Cleaned and transformed datasets produced during analysis |
| `data/samples/` | Small excerpts for development and testing |
| `notebooks/` | End-to-end experimental workflow |
| `src/config.py` | Central forecasting experiment configuration |
| `src/data/` | Dataset loading, preprocessing, and forecast-split utilities |
| `src/analysis/` | Reusable exploratory data analysis helpers |
| `src/models/` | SARIMA, LSTM, and TCN implementations (not trained in Phase 5) |
| `src/evaluation/` | Forecast evaluation metrics |
| `results/` | Experiment outputs (figures, metrics tables, predictions) |
| `docs/` | Supporting research documentation |
| `report/` | Written formative report |

## Methodology

At a high level, the planned workflow is:

1. Set up the repository and prepare external dataset access
2. Load and preprocess selected portions of the Milan dataset efficiently
3. Conduct exploratory data analysis of temporal and geographical traffic patterns
4. Review relevant literature on sequential forecasting for network traffic
5. Select three forecasting models informed by EDA and the literature
6. Run one-step-ahead forecasting experiments across geographical areas
7. Evaluate and compare models using MAE, RMSE, MAPE, and runtime
8. Analyse where and why models perform differently

**Important:** The three forecasting models are **locked**: SARIMA, LSTM, and TCN.
A naive persistence forecast is an evaluation baseline only.

## Current Status

The current stage is **Phase 6A — model implementation**. SARIMA, LSTM, TCN,
and the naive persistence baseline are implemented with smoke tests. **Final
Dec 16–22 evaluation has not been performed.**
