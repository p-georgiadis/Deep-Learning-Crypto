# src/training/model.py
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
    Attention,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError


def directional_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate directional accuracy between true and predicted values.

    Args:
        y_true: True values (batch_size, 1)
        y_pred: Predicted values (batch_size, 1)

    Returns:
        Directional accuracy score
    """
    # Ensure y_true and y_pred are 1D tensors
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))

    # Compute direction
    y_true_direction = tf.sign(y_true[1:] - y_true[:-1])
    y_pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])

    # Calculate accuracy
    accurate_directions = tf.cast(
        tf.equal(y_true_direction, y_pred_direction),
        tf.float32
    )

    return tf.reduce_mean(accurate_directions)


class CryptoPredictor:
    """
    Deep Learning model for cryptocurrency price prediction.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: List[int],
        dropout_rate: float,
        dense_units: List[int],
        attention_units: int = 64,
        learning_rate: float = 0.001,
    ):
        """
        Initialize the model.

        Args:
            input_shape: Shape of input data (sequence_length, features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            dense_units: List of units for dense layers
            attention_units: Units for the attention mechanism (default: 64)
            learning_rate: Learning rate for optimization (default: 0.001)
        """
        # Validate parameters
        self._validate_params(
            input_shape, lstm_units, dropout_rate, dense_units, attention_units, learning_rate
        )

        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.attention_units = attention_units
        self.model = None
        self.compiled = False

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Log initialized parameters
        self.logger.info(
            f"Initialized CryptoPredictor with lstm_units: {self.lstm_units}, "
            f"dense_units: {self.dense_units}, dropout_rate: {self.dropout_rate}, "
            f"learning_rate: {self.learning_rate}, attention_units: {self.attention_units}"
        )

    def _validate_params(
        self,
        input_shape: Tuple[int, int],
        lstm_units: List[int],
        dropout_rate: float,
        dense_units: List[int],
        attention_units: int,
        learning_rate: float,
    ) -> None:
        """
        Validate model parameters.

        Args:
            input_shape: Shape of input data
            lstm_units: List of LSTM units
            dropout_rate: Dropout rate
            dense_units: List of dense units
            attention_units: Attention units
            learning_rate: Learning rate

        Raises:
            ValueError: If parameters are invalid
        """
        if not all(x > 0 for x in input_shape):
            raise ValueError("Input shape dimensions must be positive")

        if not all(units > 0 for units in lstm_units):
            raise ValueError("LSTM units must be positive")

        if not all(units > 0 for units in dense_units):
            raise ValueError("Dense units must be positive")

        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")

        if attention_units <= 0:
            raise ValueError("Attention units must be positive")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

    def build(self) -> None:
        """Build simplified and optimized model architecture."""
        try:
            self.logger.info(f"Building model with input shape: {self.input_shape}")
            inputs = Input(shape=self.input_shape)
            x = inputs

            # LSTM layers with return_sequences=True for attention
            for i, units in enumerate(self.lstm_units):
                x = Bidirectional(LSTM(units, return_sequences=True))(x)
                x = BatchNormalization()(x)
                x = Dropout(self.dropout_rate)(x)
                self.logger.info(f"LSTM layer {i} output shape: {x.shape}")

            # Attention mechanism
            attention_output = Attention()([x, x])
            self.logger.info(f"Attention output shape: {attention_output.shape}")

            # Global Average Pooling to reduce sequence dimension
            x = GlobalAveragePooling1D()(attention_output)
            self.logger.info(f"GlobalAveragePooling1D output shape: {x.shape}")

            # Dense layers
            for i, units in enumerate(self.dense_units):
                x = Dense(units, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(self.dropout_rate)(x)
                self.logger.info(f"Dense layer {i} output shape: {x.shape}")

            # Single output for price prediction
            output = Dense(1, name='price_prediction')(x)
            self.logger.info(f"Output shape: {output.shape}")

            # Model assembly
            self.model = Model(inputs=inputs, outputs=output)
            self.logger.info("Model built successfully")

        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise

    def compile(self, optimizer=None, loss="mse") -> None:
        """
        Compile model with a single loss function and streamlined metrics.
        """
        try:
            if optimizer is None:
                optimizer = Adam(learning_rate=self.learning_rate)

            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=["mae", RootMeanSquaredError(), directional_accuracy],
            )

            self.compiled = True
            self.logger.info("Model compiled successfully")

        except Exception as e:
            self.logger.error(f"Error compiling model: {str(e)}")
            raise

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Tuple of (X_val, y_val)
            callbacks: List of Keras callbacks
            verbose: Verbosity level for training (default: 1)

        Returns:
            Training history
        """
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build() first.")

            # Pass all arguments to the underlying Keras model's fit method
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
            )

            self.logger.info("Model training completed")
            return history

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features

        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build() first.")

            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Evaluate the model.

        Args:
            X: Input features
            y: True values

        Returns:
            List of metric values
        """
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build() first.")

            return self.model.evaluate(X, y)

        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_history(self, history, path: Union[str, Path]) -> None:
        """
        Save training history to a JSON file.

        Args:
            history: Keras training history or dictionary
            path: Path to save the history
        """
        try:
            if isinstance(history, tf.keras.callbacks.History):
                history_dict = {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                }
            elif isinstance(history, dict):
                history_dict = history
            else:
                raise ValueError("Unsupported history object type")

            with open(path, "w") as f:
                json.dump(history_dict, f)
        except Exception as e:
            self.logger.error(f"Error saving training history: {e}")
            raise

    def save(self, path: Union[str, Path], coin_name: str) -> None:
        """
        Save the model to the specified path, including metadata.

        Args:
            path: Path to save the model.
            coin_name: Name of the coin for metadata.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        # Add metadata file
        metadata_path = Path(path).with_suffix(".json")
        metadata = {
            "coin_name": coin_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Save the model
        self.model.save(path)

    @classmethod
    def load_model_by_criteria(
        cls, models_dir: Union[str, Path], coin_name: str, latest: bool = True
    ) -> "CryptoPredictor":
        """
        Load a model matching a coin name and criteria.

        Args:
            models_dir: Directory containing models.
            coin_name: Coin name to filter models.
            latest: Whether to load the latest model.

        Returns:
            Loaded CryptoPredictor instance.
        """
        models_dir = Path(models_dir)
        metadata_files = list(models_dir.glob("*.json"))

        # Match models by metadata
        matching_models = []
        for meta_file in metadata_files:
            with open(meta_file, "r") as f:
                metadata = json.load(f)
                if metadata.get("coin_name") == coin_name:
                    matching_models.append(
                        {
                            "model_path": meta_file.with_suffix(".keras"),
                            "timestamp": metadata.get("timestamp"),
                        }
                    )

        if not matching_models:
            raise FileNotFoundError(f"No models found for coin: {coin_name}")

        # Sort by timestamp and select
        matching_models.sort(key=lambda x: x["timestamp"], reverse=latest)
        selected_model = matching_models[0]

        return cls.load(selected_model["model_path"])

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CryptoPredictor":
        """
        Load a saved model.

        Args:
            path: Path to saved model

        Returns:
            Loaded CryptoPredictor instance
        """
        try:
            keras_model = tf.keras.models.load_model(
                path,
                custom_objects={"directional_accuracy": directional_accuracy},
            )

            # Create a new CryptoPredictor instance
            input_shape = keras_model.input_shape[1:]
            predictor = cls(
                input_shape=input_shape,
                lstm_units=[],
                dropout_rate=0.0,
                dense_units=[],
                attention_units=0,
                learning_rate=0.001,
            )
            predictor.model = keras_model
            predictor.compiled = True  # Assume the model is already compiled

            return predictor

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    model = CryptoPredictor(
        input_shape=(60, 20),
        lstm_units=[64, 32],
        dropout_rate=0.2,
        dense_units=[16],
        attention_units=32,
        learning_rate=0.001,
    )

    model.build()
    model.compile()

    # Print model summary
    model.model.summary()
