"""
Forecasting model implementations.

Locked research models: SARIMA, LSTM, TCN.
Reference baseline only: Naive persistence (not a research model).

Phase 6A provides implementations and smoke tests. Final Dec 16–22 evaluation
and train/validation configuration selection belong to later phases.
"""

from src.models.lstm import LSTMConfig, LSTMModel
from src.models.naive import NaivePersistenceModel
from src.models.sarima import SARIMAConfig, SARIMAModel
from src.models.tcn import TCNConfig, TCNModel

__all__ = [
    "NaivePersistenceModel",
    "SARIMAConfig",
    "SARIMAModel",
    "LSTMConfig",
    "LSTMModel",
    "TCNConfig",
    "TCNModel",
]
