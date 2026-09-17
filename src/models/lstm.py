"""
LSTM one-step-ahead forecaster.

Input shape: (samples, 144, 1)
Output: scalar next 10-minute traffic value (scaled during training;
inverse-transformed to original units for predictions).

Uses the Phase 5 train-only MinMaxScaler — never refits the scaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EARLY_STOPPING_PATIENCE,
    LSTM_EPOCHS,
    LSTM_LAYERS,
    LSTM_LEARNING_RATE,
    LSTM_UNITS,
    PRIMARY_SEQUENCE_LENGTH,
    RANDOM_SEED,
)
from src.data.forecasting import SquareScaler
from src.models.base import (
    NeuralTrainHistory,
    TimingResult,
    assert_no_test_in_fit_window,
)
from src.models.training import (
    inverse_transform_predictions,
    predict_scaled,
    set_global_seeds,
    train_keras_regressor,
)


@dataclass
class LSTMConfig:
    """Compact, configurable LSTM architecture for fair one-step comparison."""

    sequence_length: int = PRIMARY_SEQUENCE_LENGTH
    units: int = LSTM_UNITS
    n_layers: int = LSTM_LAYERS
    dropout: float = LSTM_DROPOUT
    learning_rate: float = LSTM_LEARNING_RATE
    batch_size: int = LSTM_BATCH_SIZE
    epochs: int = LSTM_EPOCHS
    early_stopping_patience: int = LSTM_EARLY_STOPPING_PATIENCE
    seed: int = RANDOM_SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "Input → LSTM → Dense(1)",
            "sequence_length": self.sequence_length,
            "input_shape": [None, self.sequence_length, 1],
            "output_shape": [None, 1],
            "units": self.units,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "loss": "mse",
            "optimizer": "adam",
            "input_scale": "minmax_train_only",
            "early_stopping_monitor": "validation_loss_only",
        }


class LSTMModel:
    """
    Simple stacked LSTM regressor.

    Default graph:
        Input(L, 1) → LSTM(units) [× n_layers] → Dense(1)

    Validation data may be supplied for early stopping. The reserved Dec 16–22
    test period must not be used for fitting or early stopping.
    """

    name: str = "LSTM"
    is_research_model: bool = True

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        *,
        scaler: Optional[SquareScaler] = None,
    ) -> None:
        self.config = config or LSTMConfig()
        self.scaler = scaler
        self.model_: Any = None
        self.history_ = NeuralTrainHistory()
        self.timing = TimingResult()
        self._fitted = False

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def build(self) -> Any:
        """Construct a fresh Keras model (does not train)."""
        import tensorflow as tf

        set_global_seeds(self.config.seed)
        inputs = tf.keras.Input(
            shape=(self.config.sequence_length, 1), name="traffic_sequence"
        )
        x = inputs
        for i in range(self.config.n_layers):
            return_sequences = i < self.config.n_layers - 1
            x = tf.keras.layers.LSTM(
                self.config.units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )(x)
            if self.config.dropout > 0:
                x = tf.keras.layers.Dropout(self.config.dropout, name=f"dropout_{i + 1}")(x)
        outputs = tf.keras.layers.Dense(1, name="y_hat")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_one_step")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
        )
        self.model_ = model
        return model

    def summary_text(self) -> str:
        if self.model_ is None:
            self.build()
        lines: list[str] = []
        self.model_.summary(print_fn=lines.append)
        return "\n".join(lines)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_target_timestamps: Optional[np.ndarray] = None,
        val_target_timestamps: Optional[np.ndarray] = None,
        verbose: int = 0,
    ) -> "LSTMModel":
        """
        Fit on scaled sequences. Does not refit ``scaler``.

        Pass validation arrays only for early stopping / later Phase 6B selection.
        Never pass test sequences here.
        """
        if self.scaler is None:
            raise RuntimeError(
                "LSTM requires a Phase 5 SquareScaler fitted on training data only"
            )
        if train_target_timestamps is not None:
            assert_no_test_in_fit_window(train_target_timestamps, context="LSTM.fit train")
        if val_target_timestamps is not None:
            assert_no_test_in_fit_window(val_target_timestamps, context="LSTM.fit val")

        if self.model_ is None:
            self.build()

        self.model_, self.history_, fit_s, t0, t1 = train_keras_regressor(
            self.model_,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            patience=self.config.early_stopping_patience,
            verbose=verbose,
        )
        self.timing.fit_seconds = fit_s
        self.timing.fit_started_at = t0
        self.timing.fit_ended_at = t1
        self.timing.extras.update(self.history_.to_dict())
        self._fitted = True
        return self

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Call fit() before predict()")
        preds, seconds, t0, t1 = predict_scaled(self.model_, X)
        self.timing.predict_seconds = seconds
        self.timing.predict_started_at = t0
        self.timing.predict_ended_at = t1
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict in original Internet-traffic units (inverse-transformed)."""
        if self.scaler is None:
            raise RuntimeError("Scaler required for inverse-transform")
        scaled = self.predict_scaled(X)
        return inverse_transform_predictions(scaled, self.scaler)

    def input_output_shapes(self) -> dict[str, tuple]:
        return {
            "input_shape": (None, self.config.sequence_length, 1),
            "output_shape": (None, 1),
            "keras_input": (
                None
                if self.model_ is None
                else tuple(self.model_.input_shape)
            ),
            "keras_output": (
                None
                if self.model_ is None
                else tuple(self.model_.output_shape)
            ),
        }
