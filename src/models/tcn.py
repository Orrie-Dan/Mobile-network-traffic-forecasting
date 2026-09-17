"""
Temporal Convolutional Network (TCN) for one-step-ahead forecasting.

Defining properties implemented here:
- causal temporal convolution (no future leakage within the window)
- exponentially increasing dilations
- residual blocks

Input shape: (samples, 144, 1)
Output: scalar next-step prediction (scaled in training; inverse-transformed
for original-unit predictions).

Uses the Phase 5 train-only MinMaxScaler — never refits the scaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.config import (
    PRIMARY_SEQUENCE_LENGTH,
    RANDOM_SEED,
    TCN_BATCH_SIZE,
    TCN_DILATIONS,
    TCN_DROPOUT,
    TCN_EARLY_STOPPING_PATIENCE,
    TCN_EPOCHS,
    TCN_FILTERS,
    TCN_KERNEL_SIZE,
    TCN_LEARNING_RATE,
)
from src.data.forecasting import SquareScaler
from src.models.base import NeuralTrainHistory, TimingResult, assert_no_test_in_fit_window
from src.models.training import (
    inverse_transform_predictions,
    predict_scaled,
    set_global_seeds,
    train_keras_regressor,
)


@dataclass
class TCNConfig:
    """Compact causal dilated residual TCN configuration."""

    sequence_length: int = PRIMARY_SEQUENCE_LENGTH
    filters: int = TCN_FILTERS
    kernel_size: int = TCN_KERNEL_SIZE
    dilations: tuple[int, ...] = TCN_DILATIONS
    dropout: float = TCN_DROPOUT
    learning_rate: float = TCN_LEARNING_RATE
    batch_size: int = TCN_BATCH_SIZE
    epochs: int = TCN_EPOCHS
    early_stopping_patience: int = TCN_EARLY_STOPPING_PATIENCE
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if self.kernel_size < 2:
            raise ValueError("TCN kernel_size should be >= 2")
        if not self.dilations:
            raise ValueError("dilations must be non-empty")

    def receptive_field(self) -> int:
        """
        Causal receptive field at the last timestep (in steps).

        Each residual block contains **two** causal dilated convolutions with
        the same dilation ``d``. Contribution per block is ``2 * (k - 1) * d``.

        RF = 1 + sum_i 2*(k-1)*d_i
        """
        k = int(self.kernel_size)
        return 1 + 2 * (k - 1) * int(sum(self.dilations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": (
                "Input → causal dilated residual blocks → last timestep → Dense(1)"
            ),
            "sequence_length": self.sequence_length,
            "input_shape": [None, self.sequence_length, 1],
            "output_shape": [None, 1],
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "receptive_field": self.receptive_field(),
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
            "causal": True,
            "uses_dilation": True,
            "uses_residuals": True,
        }


def _causal_conv1d(
    x: Any,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    name: str,
) -> Any:
    """Left-padded causal convolution (no future frames in the receptive field)."""
    import tensorflow as tf

    # Manual left padding so Conv1D(padding='valid') is strictly causal.
    pad = (kernel_size - 1) * dilation_rate
    if pad:
        x = tf.keras.layers.ZeroPadding1D(padding=(pad, 0), name=f"{name}_pad")(x)
    return tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="valid",
        name=name,
    )(x)


def residual_tcn_block(
    x: Any,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    block_id: int,
) -> Any:
    """
    One residual TCN block:

    causal dilated Conv → ReLU → Dropout → causal dilated Conv → ReLU → Dropout
    + residual (1×1 proj if channels differ)
    """
    import tensorflow as tf

    residual = x
    y = _causal_conv1d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name=f"tcn_b{block_id}_conv1_d{dilation_rate}",
    )
    y = tf.keras.layers.Activation("relu", name=f"tcn_b{block_id}_relu1")(y)
    if dropout > 0:
        y = tf.keras.layers.SpatialDropout1D(dropout, name=f"tcn_b{block_id}_drop1")(y)

    y = _causal_conv1d(
        y,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name=f"tcn_b{block_id}_conv2_d{dilation_rate}",
    )
    y = tf.keras.layers.Activation("relu", name=f"tcn_b{block_id}_relu2")(y)
    if dropout > 0:
        y = tf.keras.layers.SpatialDropout1D(dropout, name=f"tcn_b{block_id}_drop2")(y)

    if int(residual.shape[-1]) != filters:
        residual = tf.keras.layers.Conv1D(
            filters, kernel_size=1, padding="same", name=f"tcn_b{block_id}_proj"
        )(residual)
    out = tf.keras.layers.Add(name=f"tcn_b{block_id}_add")([residual, y])
    return tf.keras.layers.Activation("relu", name=f"tcn_b{block_id}_out")(out)


class TCNModel:
    """
    Genuine TCN (Bai et al.–style causal dilated residual stack) for one-step
    traffic forecasting. Not a plain non-causal CNN.
    """

    name: str = "TCN"
    is_research_model: bool = True

    def __init__(
        self,
        config: Optional[TCNConfig] = None,
        *,
        scaler: Optional[SquareScaler] = None,
    ) -> None:
        self.config = config or TCNConfig()
        self.scaler = scaler
        self.model_: Any = None
        self.history_ = NeuralTrainHistory()
        self.timing = TimingResult()
        self._fitted = False

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def build(self) -> Any:
        import tensorflow as tf

        set_global_seeds(self.config.seed)
        inputs = tf.keras.Input(
            shape=(self.config.sequence_length, 1), name="traffic_sequence"
        )
        x = inputs
        for i, dilation in enumerate(self.config.dilations):
            x = residual_tcn_block(
                x,
                filters=self.config.filters,
                kernel_size=self.config.kernel_size,
                dilation_rate=int(dilation),
                dropout=self.config.dropout,
                block_id=i,
            )
        # Last temporal representation (causal: depends only on ≤ current time)
        x = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)
        outputs = tf.keras.layers.Dense(1, name="y_hat")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tcn_one_step")
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
    ) -> "TCNModel":
        if self.scaler is None:
            raise RuntimeError(
                "TCN requires a Phase 5 SquareScaler fitted on training data only"
            )
        if train_target_timestamps is not None:
            assert_no_test_in_fit_window(train_target_timestamps, context="TCN.fit train")
        if val_target_timestamps is not None:
            assert_no_test_in_fit_window(val_target_timestamps, context="TCN.fit val")

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
        if self.scaler is None:
            raise RuntimeError("Scaler required for inverse-transform")
        scaled = self.predict_scaled(X)
        return inverse_transform_predictions(scaled, self.scaler)

    def input_output_shapes(self) -> dict[str, tuple]:
        return {
            "input_shape": (None, self.config.sequence_length, 1),
            "output_shape": (None, 1),
            "keras_input": None if self.model_ is None else tuple(self.model_.input_shape),
            "keras_output": None if self.model_ is None else tuple(self.model_.output_shape),
            "receptive_field": self.config.receptive_field(),
        }
