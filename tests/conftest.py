# tests/conftest.py
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path


# Fixture for symbol mapping
@pytest.fixture
def symbol_mapping():
    return [
        {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT"},
        {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT"},
        {"symbol": "LTCUSDT", "baseAsset": "LTC", "quoteAsset": "USDT"},
        {"symbol": "ETHBTC", "baseAsset": "ETH", "quoteAsset": "BTC"},
        {"symbol": "LTCBTC", "baseAsset": "LTC", "quoteAsset": "BTC"}
    ]

@pytest.fixture(scope="session")
def sample_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="session")
def sample_price_history():
    """Create sample price history data"""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'open': np.random.uniform(30000, 40000, 100),
        'high': np.random.uniform(35000, 45000, 100),
        'low': np.random.uniform(25000, 35000, 100),
        'close': np.random.uniform(30000, 40000, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    }, index=dates)

@pytest.fixture(scope="session")
def sample_technical_data(sample_price_history):
    """Create sample data with technical indicators."""
    from src.preprocessing.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    return engineer.add_technical_features(sample_price_history)


@pytest.fixture(scope="session")
def sample_model_config():
    """Sample model configuration"""
    return {
        'input_shape': (60, 20),
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'dense_units': [16],
        'learning_rate': 0.001
    }

@pytest.fixture(scope="session")
def sample_training_config():
    """Sample training configuration"""
    return {
        'batch_size': 32,
        'epochs': 10,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'min_delta': 0.0001
    }

@pytest.fixture(scope="session")
def sample_sequences():
    """Create sample sequence data for model training and prediction"""
    sequence_length = 60  # Input timesteps
    features = 20         # Input features
    prediction_length = 1 # Output prediction horizon
    samples = 100         # Number of sequences

    X = np.random.random((samples, sequence_length, features))
    y = np.random.random((samples, prediction_length))
    return X, y

@pytest.fixture
def sample_predictions(sample_price_history):
    """Generate sample predictions aligned with price history."""
    prediction_length = 30  # Explicitly set the length of predictions
    indices = sample_price_history.index[-prediction_length:]
    values = np.random.uniform(30000, 40000, len(indices))
    print(f"Prediction values length: {len(values)}, Index length: {len(indices)}")  # Debugging
    return pd.Series(values, index=indices)


@pytest.fixture(scope="session")
def sample_training_history():
    """Create sample training history"""
    return {
        'loss': [0.05, 0.04, 0.035, 0.032, 0.03],
        'val_loss': [0.055, 0.045, 0.04, 0.038, 0.037],
        'mae': [0.04, 0.035, 0.03, 0.028, 0.027],
        'val_mae': [0.045, 0.04, 0.035, 0.033, 0.032]
    }

@pytest.fixture(scope="session")
def mock_api_response():
    """Mock cryptocurrency API response"""
    return {
        'prices': [[1635724800000, 61500.0], [1635811200000, 63100.0]],
        'market_caps': [[1635724800000, 1160000000000], [1635811200000, 1190000000000]],
        'total_volumes': [[1635724800000, 28000000000], [1635811200000, 31000000000]]
    }

@pytest.fixture(scope="session")
def sample_config():
    """Sample complete configuration"""
    return {
        'data': {
            'coins': ['bitcoin', 'ethereum'],
            'days': 365,
            'validation_split': 0.2,
            'test_split': 0.1,
            'symbol_mapping': [
                {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT"},
                {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT"},
                {"symbol": "LTCUSDT", "baseAsset": "LTC", "quoteAsset": "USDT"}
            ]
        },
        'model': {
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'dense_units': [16],
            'learning_rate': 0.001,
            'sequence_length': 60,
            'batch_size': 32
        },
        'training': {
            'epochs': 100,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'min_delta': 0.0001
        },
        'paths': {
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'model_to_load': 'models/trained_model.keras',
            'predict_data': 'data/predict_data.csv',
            'predictions_output': 'results/predictions.json'
        },
        'prediction': {
            'model_to_load': 'models/trained_model.keras',
            'predict_data': 'data/predict_data.csv'
        },
        'mode': 'predict'  # Ensure mode is explicitly set for the test
    }



@pytest.fixture(scope="function")
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file"""
    config_path = tmp_path / "config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)

    # Mock prediction paths to prevent validation failures
    with patch('pathlib.Path.exists', return_value=True):
        yield config_path


