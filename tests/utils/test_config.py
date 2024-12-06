# tests/test_config.py

from unittest.mock import patch

import pytest
import yaml
import os
from pathlib import Path
import tempfile


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        'data': {
            'coins': ['bitcoin', 'ethereum'],
            'days': 365,
            'validation_split': 0.2,
            'test_split': 0.1,
            'scaler': 'minmax'
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
            'processed_data_dir': 'processed_data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'results_dir': 'results',
            'visualization_dir': 'visualizations'
        },
        'prediction': {
            'model_to_load': 'models/trained_model.keras',
            'predict_data': 'data/predict_data.csv',
            'predictions_output': 'results/predictions.json'
        }
    }


@pytest.fixture
def config_file(tmp_path, sample_config):
    """Create a temporary config file"""
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


def test_config_initialization(config_file):
    """Test configuration initialization"""
    from src.utils.config import Config

    with patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config = Config(config_file)
        config.validate_config(mode='train')  # Explicit mode
        assert hasattr(config, 'config')
        assert isinstance(config.config, dict)
        assert 'data' in config.config
        assert 'model' in config.config
        assert 'training' in config.config
        assert 'prediction' in config.config

def test_config_validation(sample_config):
    """Test configuration validation"""
    from src.utils.config import Config

    with tempfile.TemporaryDirectory() as tmp_dir, patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config_path = Path(tmp_dir) / "config.yaml"

        # Test with valid config
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        config = Config(config_path)
        assert config.validate_config(mode='train')

        # Test with missing required fields
        invalid_config = sample_config.copy()
        del invalid_config['model']

        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError):
            Config(config_path)


def test_get_prediction_config(config_file):
    """Test getting prediction configuration"""
    from src.utils.config import Config

    with patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config = Config(config_file)
        prediction_config = config.config.get('prediction')

        assert isinstance(prediction_config, dict)
        assert 'model_to_load' in prediction_config
        assert 'predict_data' in prediction_config
        assert 'predictions_output' in prediction_config


def test_validate_prediction_config(sample_config):
    """Test validation of prediction configuration"""
    from src.utils.config import Config

    valid_config = sample_config.copy()
    valid_config['prediction']['model_to_load'] = '/does/not/exist/model.keras'

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"

        # Write invalid config
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)

        config = Config(config_path)
        with pytest.raises(ValueError, match="prediction.model_to_load path does not exist"):
            config.validate_config(mode='predict')  # Explicit mode


def test_update_config(config_file):
    """Test configuration updating"""
    from src.utils.config import Config

    with patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config = Config(config_file)

        # Update existing value
        config.update_config('model.lstm_units', [128, 64])
        assert config.config['model']['lstm_units'] == [128, 64]

        # Add new value
        config.update_config('model.new_param', 'value')
        assert config.config['model']['new_param'] == 'value'

        # Test nested update
        config.update_config('data.preprocessing.scaler', 'standard')
        assert config.config['data']['preprocessing']['scaler'] == 'standard'


def test_save_config(tmp_path, config_file):
    """Test configuration saving"""
    from src.utils.config import Config

    with patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config = Config(config_file)
        save_path = tmp_path / "saved_config.yaml"

        # Update and save
        config.update_config('model.lstm_units', [128, 64])
        config.save_config(save_path)

        # Load saved config and verify
        new_config = Config(save_path)
        assert new_config.config['model']['lstm_units'] == [128, 64]


def test_config_environment_variables(config_file):
    """Test environment variable integration"""
    from src.utils.config import Config

    # Set environment variables
    os.environ['CRYPTO_MODEL_LR'] = '0.01'

    with patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config = Config(config_file)
        assert config.get_env_value('CRYPTO_MODEL_LR', default=0.001) == 0.01

        # Test with non-existent variable
        assert config.get_env_value('NON_EXISTENT_VAR', default='default') == 'default'


def test_config_range_validation(config_file):
    """Test configuration range validation"""
    from src.utils.config import Config

    with patch('pathlib.Path.exists', return_value=True):  # Mock file existence
        config = Config(config_file)

        # Test invalid ranges
        with pytest.raises(ValueError):
            config.update_config('model.dropout_rate', 1.5)  # Should be between 0 and 1

        with pytest.raises(ValueError):
            config.update_config('training.epochs', -1)  # Should be positive

        with pytest.raises(ValueError):
            config.update_config('data.validation_split', 1.2)  # Should be between 0 and 1

