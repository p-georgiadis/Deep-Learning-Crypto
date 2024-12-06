# tests/training/test_trainer.py
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import json


@pytest.fixture
def sample_training_data():
    """Create sample data for training"""
    X_train = np.random.random((1000, 60, 20))  # 1000 samples, 60 timesteps, 20 features
    y_train = np.random.random((1000, 1))
    X_val = np.random.random((200, 60, 20))
    y_val = np.random.random((200, 1))

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }


@pytest.fixture
def trainer_config():
    """Sample trainer configuration"""
    return {
        'batch_size': 32,
        'epochs': 10,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'model_dir': 'test_models'
    }


@pytest.fixture
def model_config():
    """Sample model configuration"""
    return {
        'input_shape': (60, 20),
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'dense_units': [16],
        'learning_rate': 0.001
    }


def test_trainer_initialization(trainer_config, model_config):
    """Test trainer initialization"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    trainer = ModelTrainer(model, **trainer_config)

    assert trainer.batch_size == trainer_config['batch_size']
    assert trainer.epochs == trainer_config['epochs']
    assert trainer.model_dir == Path(trainer_config['model_dir'])
    assert trainer.model is not None


def test_prepare_callbacks(tmp_path, trainer_config, model_config):
    """Test callback preparation"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    trainer_config['model_dir'] = tmp_path
    model = CryptoPredictor(**model_config)
    trainer = ModelTrainer(model, **trainer_config)

    callbacks = trainer.prepare_callbacks()

    # Check if essential callbacks are present
    callback_types = [type(callback).__name__ for callback in callbacks]
    assert 'EarlyStopping' in callback_types
    assert 'ModelCheckpoint' in callback_types
    assert 'ReduceLROnPlateau' in callback_types
    assert 'CustomTensorBoard' in callback_types


def test_train_model(sample_training_data, trainer_config, model_config):
    """Test model training"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_config['model_dir'] = tmp_dir
        model = CryptoPredictor(**model_config)

        # Build and compile the model
        model.build()
        model.compile()

        trainer = ModelTrainer(model, **trainer_config)

        # Train the model
        history = trainer.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            sample_training_data['X_val'],
            sample_training_data['y_val']
        )

        # Verify training history
        assert isinstance(history.history, dict)
        assert 'loss' in history.history
        assert len(history.history['loss']) > 0


def test_save_load_model(sample_training_data, trainer_config, model_config):
    """Test model saving and loading"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_config['model_dir'] = tmp_dir
        model = CryptoPredictor(**model_config)

        # Build and compile the model
        model.build()
        model.compile()

        trainer = ModelTrainer(model, **trainer_config)

        # Train the model
        trainer.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            sample_training_data['X_val'],
            sample_training_data['y_val']
        )

        # Save the model
        save_path = Path(tmp_dir) / 'model.keras'
        trainer.save_model(save_path)

        # Load and verify the model
        loaded_model = CryptoPredictor.load(save_path)
        assert loaded_model.model is not None


def test_model_evaluation(sample_training_data, trainer_config, model_config):
    """Test model evaluation"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()
    trainer = ModelTrainer(model, **trainer_config)

    # Train the model first
    trainer.train(
        sample_training_data['X_train'],
        sample_training_data['y_train'],
        sample_training_data['X_val'],
        sample_training_data['y_val']
    )

    # Evaluate
    metrics = trainer.evaluate(
        sample_training_data['X_val'],
        sample_training_data['y_val']
    )

    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_prediction_generation(sample_training_data, trainer_config, model_config):
    """Test prediction generation"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    model = CryptoPredictor(**model_config)
    model.build()
    model.compile()
    trainer = ModelTrainer(model, **trainer_config)

    # Train the model
    trainer.train(
        sample_training_data['X_train'],
        sample_training_data['y_train'],
        sample_training_data['X_val'],
        sample_training_data['y_val']
    )

    # Generate predictions
    predictions = trainer.predict(sample_training_data['X_val'])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(sample_training_data['X_val']), 1)
    assert not np.any(np.isnan(predictions))


def test_custom_callbacks(trainer_config, model_config):
    """Test adding custom callbacks"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pass

    model = CryptoPredictor(**model_config)
    trainer = ModelTrainer(model, **trainer_config)

    # Add custom callback
    custom_callback = CustomCallback()
    trainer.add_callback(custom_callback)

    callbacks = trainer.prepare_callbacks()
    assert any(isinstance(cb, CustomCallback) for cb in callbacks)


def test_learning_rate_schedule(sample_training_data, trainer_config, model_config):
    """Test learning rate scheduling"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_config['model_dir'] = tmp_dir
        model = CryptoPredictor(**model_config)
        model.build()
        model.compile()
        trainer = ModelTrainer(model, **trainer_config)

        # Train model
        history = trainer.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            sample_training_data['X_val'],
            sample_training_data['y_val']
        )

        # Check if learning rate was adjusted
        assert 'lr' in history.history


def test_save_load_model(sample_training_data, trainer_config, model_config):
    """Test model saving and loading"""
    from src.training.trainer import ModelTrainer
    from src.training.model import CryptoPredictor

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_config['model_dir'] = tmp_dir
        model = CryptoPredictor(**model_config)
        model.build()
        model.compile()
        trainer = ModelTrainer(model, **trainer_config)

        # Train and save model
        trainer.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            sample_training_data['X_val'],
            sample_training_data['y_val']
        )

        # Use a valid extension for the saved model
        save_path = Path(tmp_dir) / 'final_model.keras'
        trainer.save_model(save_path)

        # Load and verify the model
        loaded_predictor = CryptoPredictor.load(save_path)
        assert loaded_predictor.model is not None

        # Ensure predictions are consistent
        original_preds = trainer.predict(sample_training_data['X_val'])
        loaded_preds = loaded_predictor.predict(sample_training_data['X_val'])
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

