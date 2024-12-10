# src/training/model.py

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Callable
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from src.utils.custom_losses import di_mse_loss, directional_accuracy



class CryptoPredictor:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        dense_units: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        clip_norm: Optional[float] = None,
    ):
        self.logger = self._setup_logger()

        self.input_shape = input_shape
        self.lstm_units = lstm_units or [64]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units or [32]
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm

        self.model = None
        self.compiled = False

        self.logger.info(
            f"CryptoPredictor init: LSTM units={self.lstm_units}, "
            f"Dense units={self.dense_units}, Dropout={self.dropout_rate}, "
            f"LR={self.learning_rate}, Clip Norm={self.clip_norm}"
        )

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def build(self) -> None:
        self.logger.info("Building model architecture...")
        inputs = Input(shape=self.input_shape)
        x = inputs

        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = Bidirectional(LSTM(units, return_sequences=return_sequences))(x)
            x = Dropout(self.dropout_rate)(x)

        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation="relu")(x)
            x = Dropout(self.dropout_rate)(x)

        output = Dense(1, name="price_prediction")(x)

        self.model = tf.keras.Model(inputs, output)
        self.logger.info("Model architecture built successfully.")

    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[Union[str, Callable]] = None,
    ) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        if optimizer is None:
            if self.clip_norm:
                optimizer = Adam(learning_rate=self.learning_rate, clipnorm=self.clip_norm)
            else:
                optimizer = Adam(learning_rate=self.learning_rate)
            self.logger.info("Using default Adam optimizer.")

        if loss is None:
            loss = di_mse_loss
            self.logger.info("Using DI-MSE loss for direction-integrated regression.")

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["mae", RootMeanSquaredError(name="rmse"), directional_accuracy]
        )
        self.compiled = True
        self.logger.info("Model compiled successfully.")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        if self.model is None or not self.compiled:
            raise ValueError("Model not ready. Ensure it's built and compiled.")

        self.logger.info(f"Training model for {epochs} epochs and batch size {batch_size}.")
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        self.logger.info("Training complete.")
        return history

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        self.logger.info("Generating predictions...")
        preds = self.model.predict(X, verbose=verbose)
        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: int = 0) -> Dict[str, float]:
        if self.model is None or not self.compiled:
            raise ValueError("Model not ready. Ensure it's built and compiled.")

        self.logger.info("Evaluating model...")
        results = self.model.evaluate(X, y, verbose=verbose)
        metrics_names = ["loss", "mae", "rmse", "directional_accuracy"]
        metrics = {name: float(val) for name, val in zip(metrics_names, results)}
        self.logger.info(f"Evaluation results: {metrics}")
        return metrics

    def save_history(self, history: Union[tf.keras.callbacks.History, Dict], path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(history, tf.keras.callbacks.History):
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        else:
            history_dict = history

        with open(path, "w") as f:
            json.dump(history_dict, f, indent=4)

        self.logger.info(f"History saved to {path}.")

    def save(self, path: Union[str, Path]) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.logger.info(f"Model saved to {path}.")

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "CryptoPredictor":
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}.")

        keras_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "directional_accuracy": directional_accuracy,
                "di_mse_loss": di_mse_loss
            }
        )
        input_shape = keras_model.input_shape[1:]
        predictor = cls(input_shape=input_shape)
        predictor.model = keras_model
        predictor.compiled = True
        logging.info(f"Loaded model from {model_path}")
        return predictor
