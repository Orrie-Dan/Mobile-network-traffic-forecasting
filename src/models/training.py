"""
Neural training utilities shared by LSTM and TCN.

Early stopping, if used, monitors validation loss only — never test loss.
Scalers must already be fitted on training data (Phase 5); they are never refit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import MODEL_CHECKPOINT_DIR, RANDOM_SEED
from src.data.forecasting import SquareScaler
from src.models.base import NeuralTrainHistory, Timer, reshape_sequences_for_keras


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Best-effort reproducibility for NumPy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # noqa: BLE001
        pass


def ensure_checkpoint_dir(root: Optional[Path] = None) -> Path:
    """Create a gitignored temporary checkpoint directory."""
    base = root or Path(__file__).resolve().parents[2]
    path = base / MODEL_CHECKPOINT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_keras_regressor(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int,
    batch_size: int,
    patience: int,
    verbose: int = 0,
) -> tuple[Any, NeuralTrainHistory, float, str, str]:
    """
    Fit a Keras model with optional early stopping on validation loss.

    Returns (model, history, fit_seconds, fit_started_at, fit_ended_at).
    Test data must not be passed as validation.
    """
    import tensorflow as tf

    X_tr = reshape_sequences_for_keras(X_train)
    y_tr = np.asarray(y_train, dtype="float32").reshape(-1, 1)
    if len(X_tr) != len(y_tr):
        raise ValueError("X_train and y_train length mismatch")

    callbacks: list[Any] = []
    validation_data = None
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("Provide both X_val and y_val, or neither")
        X_v = reshape_sequences_for_keras(X_val)
        y_v = np.asarray(y_val, dtype="float32").reshape(-1, 1)
        validation_data = (X_v, y_v)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            )
        )

    timer = Timer().start()
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    fit_seconds = timer.stop()

    history = NeuralTrainHistory(
        epochs_completed=int(len(hist.history.get("loss", []))),
        train_loss=[float(v) for v in hist.history.get("loss", [])],
        validation_loss=[float(v) for v in hist.history.get("val_loss", [])],
        stopped_early=bool(
            validation_data is not None
            and len(hist.history.get("loss", [])) < epochs
        ),
    )
    if history.validation_loss:
        history.best_epoch = int(np.argmin(history.validation_loss)) + 1
    return model, history, fit_seconds, timer.started_at or "", timer.ended_at or ""


def predict_scaled(
    model: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, float, str, str]:
    """Predict in scaled space. Returns (y_hat_scaled, seconds, start, end)."""
    X_in = reshape_sequences_for_keras(X)
    timer = Timer().start()
    preds = model.predict(X_in, verbose=0)
    seconds = timer.stop()
    arr = np.asarray(preds, dtype="float64").reshape(-1)
    return arr, seconds, timer.started_at or "", timer.ended_at or ""


def inverse_transform_predictions(
    y_scaled: np.ndarray,
    scaler: SquareScaler,
) -> np.ndarray:
    """
    Map scaled predictions back to original Internet-traffic units.

    The scaler must already be fitted on training data only (Phase 5).
    """
    if not hasattr(scaler.scaler, "n_samples_seen_"):
        raise RuntimeError("Scaler is not fitted; do not fit scalers inside model code")
    return scaler.inverse_transform(np.asarray(y_scaled, dtype="float64"))
