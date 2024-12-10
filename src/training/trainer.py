# src/training/trainer.py

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


class ModelTrainer:
    """
    A simplified trainer class that handles model training and saving training history.
    Assumes that:
    - The model is already built and compiled.
    - Data (X_train, y_train, X_val, y_val) is already prepared and scaled.
    - y now has shape (samples, 2): y[:,0] is current day's actual price,
      y[:,1] is previous day's actual price (for direction checking).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_dir: Union[str, Path],
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        min_delta: float = 0.0001,
    ):
        """
        Initialize the trainer with given training parameters.

        Args:
            model: A compiled Keras model ready to be trained.
            model_dir: Directory to save model checkpoints and history.
            batch_size: Training batch size.
            epochs: Number of epochs.
            early_stopping_patience: Patience for EarlyStopping callback.
            min_delta: Minimum delta for EarlyStopping.
        """
        self.model = model
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(
            f"ModelTrainer initialized with batch_size={self.batch_size}, "
            f"epochs={self.epochs}, early_stopping_patience={self.early_stopping_patience}, "
            f"min_delta={self.min_delta}, model_dir={self.model_dir}"
        )

    def validate_data_shape(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input shapes for training/validation data.

        We now expect y to have shape (samples, 2):
        y[:,0]: current day's actual price
        y[:,1]: previous day's actual price

        Args:
            X: Input feature array, expected shape (samples, timesteps, features)
            y: Target array, expected shape (samples, 2)
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")

        if X.ndim != 3:
            raise ValueError(f"Expected X of shape (samples, timesteps, features), got {X.shape}.")

        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"Expected y of shape (samples, 2), got {y.shape}.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        self.logger.info(f"Data shapes are valid: X={X.shape}, y={y.shape}")

    def prepare_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Prepare minimal set of callbacks: ModelCheckpoint and EarlyStopping.

        Returns:
            List of callbacks.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            callbacks: List[tf.keras.callbacks.Callback] = []

            # Use .keras format for model checkpoints
            checkpoint_path = self.model_dir / f"model_checkpoint_{timestamp}.keras"

            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='min'
            ))
            self.logger.info(f"Added ModelCheckpoint callback: {checkpoint_path}")

            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=self.min_delta,
                verbose=1,
                restore_best_weights=True
            ))
            self.logger.info(f"Added EarlyStopping callback with patience {self.early_stopping_patience}")

            return callbacks

        except Exception as e:
            self.logger.error(f"Error preparing callbacks: {str(e)}")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        additional_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets (shape (samples,2))
            X_val: Validation features.
            y_val: Validation targets (shape (samples,2))
            additional_callbacks: Optionally add more callbacks.

        Returns:
            Keras History object.
        """
        self.validate_data_shape(X_train, y_train)
        self.validate_data_shape(X_val, y_val)

        self.logger.info(f"Starting training for {self.epochs} epochs with batch size {self.batch_size}.")

        callbacks = self.prepare_callbacks()
        if additional_callbacks:
            callbacks.extend(additional_callbacks)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.logger.info("Training completed.")
        self.save_history(history)
        return history

    def save_history(self, history: tf.keras.callbacks.History) -> None:
        """
        Save training history to a JSON file in the model directory.
        """
        history_path = self.model_dir / "training_history.json"
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        self.logger.info(f"Training history saved to {history_path}.")
