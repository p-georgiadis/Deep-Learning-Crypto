# src/training/trainer.py
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard



class CustomTensorBoard(TensorBoard):
    """Extended TensorBoard callback with additional metrics logging."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """Log additional metrics at the end of each epoch."""
        logs = logs or {}
        # Log current learning rate
        if self.model.optimizer and hasattr(self.model.optimizer, 'learning_rate'):
            logs['lr'] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        super().on_epoch_end(epoch, logs)

class ModelTrainer:
    """
    Handles model training, evaluation, and prediction.
    This trainer assumes `self.model` is an instance of `CryptoPredictor`,
    which has .fit, .evaluate, .predict, and .save methods.
    """

    def __init__(
            self,
            model,
            config,
            batch_size: int,
            epochs: int,
            early_stopping_patience: int,
            reduce_lr_patience: int,
            min_delta: float,
            model_dir: Union[str, Path]
    ):
        """
        Initialize the trainer.

        Args:
            model: Instance of CryptoPredictor
            config: Config object that contains configuration dictionaries
            batch_size: Training batch size
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            min_delta: Minimum change in monitored value for early stopping
            model_dir: Directory to save model artifacts
        """
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.min_delta = min_delta
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Log initialized parameters
        self.logger.info(f"Initialized ModelTrainer with batch_size: {self.batch_size}, "
                         f"epochs: {self.epochs}, early_stopping_patience: {self.early_stopping_patience}, "
                         f"reduce_lr_patience: {self.reduce_lr_patience}, min_delta: {self.min_delta}")

        self.callbacks = []

    def prepare_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Prepare training callbacks.

        Returns:
            List of Keras callbacks
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            callbacks = []

            # Paths from config
            logs_dir = Path(self.config.config['paths']['model_logs_dir']) / timestamp

            # Model checkpoint callback (save best model based on val_loss)
            checkpoint_path = self.model_dir / f"model_checkpoint_{timestamp}.keras"
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='min'
            )
            callbacks.append(checkpoint)

            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=self.min_delta,
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

            # Reduce LR on Plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.reduce_lr_patience,
                min_delta=self.min_delta,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)

            # TensorBoard callback
            tensorboard = CustomTensorBoard(
                log_dir=str(logs_dir),
                histogram_freq=0,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=0  # Disable profiling
            )
            callbacks.append(tensorboard)

            # Include any callbacks added manually
            callbacks.extend(self.callbacks)

            return callbacks

        except Exception as e:
            self.logger.error(f"Error preparing callbacks: {str(e)}")
            raise

    def add_callback(self, callback: tf.keras.callbacks.Callback) -> None:
        """
        Add a custom callback.

        Args:
            callback: Keras callback instance
        """
        self.callbacks.append(callback)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            custom_callbacks: Additional callbacks to add

        Returns:
            Training history
        """
        # Clear session before training (optional)
        tf.keras.backend.clear_session()

        try:
            self.logger.info("Starting model training")
            # Basic sanity checks on data
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                raise ValueError("X_train contains NaNs or Infs.")
            if np.isnan(y_train).any() or np.isinf(y_train).any():
                raise ValueError("y_train contains NaNs or Infs.")

            # Check if model is compiled
            if not self.model.compiled:
                raise ValueError("Model not compiled. Call model.compile() before train().")

            # Prepare callbacks
            callbacks = self.prepare_callbacks()
            if custom_callbacks:
                callbacks.extend(custom_callbacks)

            # Train the model using CryptoPredictor's fit method
            history = self.model.fit(
                X_train=X_train,
                y_train=y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            self.logger.info("Model training completed")

            # Save training history
            self.save_history(
                history=history,
                path=self.model_dir / "training_history.json"
            )

            return history

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model using the compiled metrics from CryptoPredictor.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics including loss and any compiled metrics
        """
        try:
            self.logger.info("Evaluating model")
            # Evaluate with CryptoPredictor's evaluate method
            results = self.model.evaluate(X_test, y_test)
            # results should be [loss, mae, rmse, directional_accuracy] if compiled as so

            # Convert results to a dictionary
            # The order is: loss (from compile), mae, root_mean_squared_error, directional_accuracy (if included)
            metrics = {}
            # If we set metrics=["mae", RootMeanSquaredError(), directional_accuracy] in model.compile
            # The order is: loss, mae, root_mean_squared_error, directional_accuracy
            if len(results) == 4:
                loss, mae, rmse, da = results
                metrics = {
                    "loss": float(loss),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "directional_accuracy": float(da)
                }
            else:
                # If for some reason we have a different metrics configuration, adjust accordingly
                self.logger.warning("Unexpected number of metrics returned. Check model.compile settings.")
                # Just do a generic mapping
                # first is always loss
                metrics["loss"] = float(results[0])
                if len(results) > 1:
                    metrics["mae"] = float(results[1])
                if len(results) > 2:
                    metrics["rmse"] = float(results[2])
                if len(results) > 3:
                    metrics["directional_accuracy"] = float(results[3])

            self.logger.info(f"Evaluation Metrics: {metrics}")

            # Save metrics to a JSON file
            metrics_path = self.model_dir / "evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            return metrics

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
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
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise

    def save_model(self, path: Union[str, Path], coin_name: str) -> None:
        """
        Save the trained model using CryptoPredictor's save method.

        Args:
            path: Save path (e.g. models/model_name.keras)
            coin_name: Name of the coin for metadata
        """
        try:
            self.model.save(path, coin_name)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def save_history(self, history: Union[tf.keras.callbacks.History, Dict], path: Path) -> None:
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

            with open(path, 'w') as f:
                json.dump(history_dict, f, indent=4)
            self.logger.info(f"Training history saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving training history: {e}")
            raise


if __name__ == "__main__":
    import numpy as np
    import logging
    import sys
    import tensorflow as tf
    from pathlib import Path
    from tqdm import tqdm

    # Set logging level for debugging
    logging.getLogger().setLevel(logging.DEBUG)
    print("Debug run starting...")

    # Mock configuration object that mirrors your main config structure
    class MockConfig:
        def __init__(self):
            self.config = {
                'paths': {
                    'model_logs_dir': 'logs_debug'
                }
            }
        def get_logging_config(self):
            return {
                'name': 'crypto_predictor',
                'log_dir': 'logs_debug',
                'console_level': 'DEBUG',
                'file_level': 'DEBUG',
                'rotation': 'time',
                'json_format': False
            }
        def get_model_config(self):
            return {
                'sequence_length': 60,
                'prediction_length': 1,
                'lstm_units': [32, 16],
                'dropout_rate': 0.2,
                'dense_units': [32],
                'attention_units': 32,
                'learning_rate': 0.001,
                'regularization': 0.0001
            }
        def get_training_config(self):
            return {
                'batch_size': 32,
                'epochs': 2,  # fewer epochs for testing
                'early_stopping_patience': 3,
                'reduce_lr_patience': 2,
                'min_delta': 0.0001,
                'checkpoint_frequency': 0
            }

    config = MockConfig()

    # Synthetic dataset
    X_train = np.random.randn(1024, 60, 20).astype(np.float32)
    y_train = np.random.randn(1024, 1).astype(np.float32)
    X_val = np.random.randn(128, 60, 20).astype(np.float32)
    y_val = np.random.randn(128, 1).astype(np.float32)

    # Import your CryptoPredictor from model.py
    from src.training.model import CryptoPredictor

    # Create the model with the same config as main.py
    mc = config.get_model_config()
    model = CryptoPredictor(
        input_shape=(mc['sequence_length'], X_train.shape[2]),
        lstm_units=mc['lstm_units'],
        dropout_rate=mc['dropout_rate'],
        dense_units=mc['dense_units'],
        learning_rate=mc['learning_rate'],
        attention_units=mc['attention_units']
    )
    model.build()
    model.compile()

    # Import ModelTrainer
    from src.training.trainer import ModelTrainer

    # Add debugging callbacks
    class BatchDebugCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            print(f"[DEBUG] Batch {batch} ended. logs={logs}")

        def on_epoch_end(self, epoch, logs=None):
            print(f"[DEBUG] Epoch {epoch} ended. logs={logs}")

    # Re-define ProgressCallback here (from main.py)
    class ProgressCallback(tf.keras.callbacks.Callback):
        """Custom callback for a training progress bar."""
        def __init__(self, epochs):
            super().__init__()
            self.progress_bar = tqdm(total=epochs, desc="Training", unit="epoch", position=0, leave=True)

        def on_epoch_end(self, epoch, logs=None):
            self.progress_bar.update(1)
            logs = logs or {}
            self.progress_bar.set_postfix({
                'loss': f"{logs.get('loss', 0):.4f}",
                'val_loss': f"{logs.get('val_loss', 0):.4f}"
            })

        def on_train_end(self, logs=None):
            self.progress_bar.close()

    # Create trainer
    tc = config.get_training_config()
    trainer = ModelTrainer(
        model=model,
        config=config,
        batch_size=tc['batch_size'],
        epochs=tc['epochs'],
        early_stopping_patience=tc['early_stopping_patience'],
        reduce_lr_patience=tc['reduce_lr_patience'],
        min_delta=tc['min_delta'],
        model_dir='models_debug'
    )

    # Add callbacks similar to main.py environment
    trainer.add_callback(BatchDebugCallback())
    trainer.add_callback(ProgressCallback(epochs=tc['epochs']))

    print("Starting training (debugging run)...")

    try:
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        print("[DEBUG] Training completed. History keys:", history.history.keys())
        # Evaluate
        metrics = trainer.evaluate(X_val, y_val)
        print("[DEBUG] Evaluation metrics:", metrics)
        # Predictions
        preds = trainer.predict(X_val[:5])
        print("[DEBUG] Sample predictions:", preds)
    except Exception as e:
        print("[ERROR] Training failed with exception:", e)

    print("Debug run completed successfully (if we got here without freezing).")
