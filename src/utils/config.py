# src/utils/config.py
import logging
import os
from pathlib import Path
from typing import Dict, Any, Union

import yaml


class Config:
    """
    Configuration management for the cryptocurrency prediction project.

    Handles:
    - Loading configuration from YAML
    - Environment variable integration
    - Configuration validation
    - Dynamic updates
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)

        # Load and validate configuration
        self.config = self._load_config()
        self.validate_config()

        # Create required directories
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update with environment variables
            config = self._update_from_env(config)
            return config

        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _update_from_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with environment variables.

        Args:
            config: Base configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        env_mapping = {
            'CRYPTO_MODEL_LR': ('model', 'learning_rate'),
            'CRYPTO_BATCH_SIZE': ('training', 'batch_size'),
            'CRYPTO_EPOCHS': ('training', 'epochs'),
            'CRYPTO_DATA_DIR': ('paths', 'data_dir')
        }

        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        raise ValueError(f"Configuration section '{key}' missing when updating from environment variables.")
                    current = current[key]
                current[config_path[-1]] = self._convert_type(os.environ[env_var])

        return config

    def _convert_type(self, value: str) -> Union[int, float, str]:
        """
        Convert string value to appropriate type.

        Args:
            value: String value to convert

        Returns:
            Converted value
        """
        try:
            # Try converting to int
            return int(value)
        except ValueError:
            try:
                # Try converting to float
                return float(value)
            except ValueError:
                # Return as string
                return value

    def validate_config(self, mode: str = None) -> bool:
        """
        Validate configuration structure and values.

        Args:
            mode (str): The pipeline mode for validation (optional).

        Returns:
            bool: True if configuration is valid.

        Raises:
            ValueError: If configuration is invalid.
        """
        required_sections = ['data', 'model', 'training', 'paths']

        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

        # Validate individual sections
        self._validate_data_config()
        self._validate_model_config()
        self._validate_training_config()
        self._validate_logging_config()

        # Validate prediction configuration only if mode is provided
        if mode:
            self._validate_prediction_config(mode)

        return True

    def _validate_data_config(self) -> None:
        """Validate data configuration section."""
        data_config = self.config['data']

        # Validation for validation_split
        if 'validation_split' in data_config:
            val_split = data_config['validation_split']
            if not 0 < val_split < 1:
                raise ValueError("data.validation_split must be between 0 and 1")

        # Validation for test_split
        if 'test_split' in data_config:
            test_split = data_config['test_split']
            if not 0 < test_split < 1:
                raise ValueError("data.test_split must be between 0 and 1")

        # Ensure validation_split and test_split do not overlap excessively
        if 'validation_split' in data_config and 'test_split' in data_config:
            if data_config['validation_split'] + data_config['test_split'] >= 1:
                raise ValueError("The sum of validation_split and test_split must be less than 1")

        # Validation for scaler
        if 'scaler' in data_config and data_config['scaler'] not in ['minmax', 'standard']:
            raise ValueError("data.scaler must be 'minmax' or 'standard'")

        # Validation for coins
        if 'coins' in data_config:
            if not isinstance(data_config['coins'], list) or not all(
                    isinstance(coin, str) for coin in data_config['coins']):
                raise ValueError("data.coins must be a list of strings")

        # Validation for days
        if 'days' in data_config:
            if not isinstance(data_config['days'], int) or data_config['days'] <= 0:
                raise ValueError("data.days must be a positive integer")

    def _validate_model_config(self) -> None:
        """Validate model configuration section."""
        model_config = self.config['model']

        # Validate dropout_rate
        if 'dropout_rate' in model_config:
            dropout = model_config['dropout_rate']
            if not 0 <= dropout < 1:
                raise ValueError("model.dropout_rate must be between 0 and 1")

        # Validate sequence_length
        if 'sequence_length' in model_config:
            seq_length = model_config['sequence_length']
            if not isinstance(seq_length, int) or seq_length <= 0:
                raise ValueError("model.sequence_length must be a positive integer")

        # Validate lstm_units
        if 'lstm_units' in model_config:
            if not isinstance(model_config['lstm_units'], list) or not all(
                    isinstance(unit, int) and unit > 0 for unit in model_config['lstm_units']):
                raise ValueError("model.lstm_units must be a list of positive integers")

        # Validate dense_units
        if 'dense_units' in model_config:
            if not isinstance(model_config['dense_units'], list) or not all(
                    isinstance(unit, int) and unit > 0 for unit in model_config['dense_units']):
                raise ValueError("model.dense_units must be a list of positive integers")

        # Validate learning_rate
        if 'learning_rate' in model_config:
            learning_rate = model_config['learning_rate']
            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError("model.learning_rate must be a positive number")

    def _validate_training_config(self) -> None:
        """Validate training configuration section."""
        training_config = self.config['training']

        # Validate epochs
        if 'epochs' in training_config:
            epochs = training_config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError("training.epochs must be a positive integer")

        # Validate batch_size
        if 'batch_size' in training_config:
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("training.batch_size must be a positive integer")

        # Validate early_stopping_patience
        if 'early_stopping_patience' in training_config:
            patience = training_config['early_stopping_patience']
            if not isinstance(patience, int) or patience < 0:
                raise ValueError("training.early_stopping_patience must be a non-negative integer")

        # Validate reduce_lr_patience
        if 'reduce_lr_patience' in training_config:
            reduce_patience = training_config['reduce_lr_patience']
            if not isinstance(reduce_patience, int) or reduce_patience < 0:
                raise ValueError("training.reduce_lr_patience must be a non-negative integer")

        # Validate min_delta
        if 'min_delta' in training_config:
            min_delta = training_config['min_delta']
            if not isinstance(min_delta, (float, int)) or min_delta < 0:
                raise ValueError("training.min_delta must be a non-negative number")

    def _validate_prediction_config(self, mode: str):
        """Validate prediction configuration section."""
        prediction_config = self.config.get('prediction', {})

        # Validate model_to_load path
        if mode in ['predict', 'full-pipeline'] and 'model_to_load' in prediction_config:
            model_path = prediction_config['model_to_load']
            if not Path(model_path).exists():
                raise ValueError(f"prediction.model_to_load path does not exist: {model_path}")

        # Validate predict_data path
        if mode in ['predict', 'full-pipeline'] and 'predict_data' in prediction_config:
            data_path = prediction_config['predict_data']
            if not Path(data_path).exists():
                raise ValueError(f"prediction.predict_data path does not exist: {data_path}")

    def _validate_logging_config(self) -> None:
        """Validate logging configuration section."""
        logging_config = self.config.get('logging', {})

        required_keys = ['name', 'log_dir', 'console_level', 'file_level', 'rotation', 'json_format']
        for key in required_keys:
            if key not in logging_config:
                raise ValueError(f"Missing required logging config parameter: {key}")

        # Validate log levels
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if logging_config['console_level'] not in valid_levels:
            raise ValueError("logging.console_level must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")
        if logging_config['file_level'] not in valid_levels:
            raise ValueError("logging.file_level must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

        # Validate rotation
        if logging_config['rotation'] not in ['time', 'size']:
            raise ValueError("logging.rotation must be either 'time' or 'size'")

        # Validate json_format
        if not isinstance(logging_config['json_format'], bool):
            raise ValueError("logging.json_format must be a boolean")

        # Validate that log_dir is a valid path
        if not isinstance(logging_config['log_dir'], str):
            raise ValueError("logging.log_dir must be a string representing a directory path")

    def _create_directories(self) -> None:
        """Create required directories from configuration."""
        paths = self.config.get('paths', {})
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config.get('training', {})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config.get('data', {})

    def update_config(self, path: str, value: Any) -> None:
        """
        Update configuration value.

        Args:
            path: Dot-separated path to config value
            value: New value
        """
        try:
            # Split path into parts
            parts = path.split('.')

            # Navigate to the correct position
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    raise ValueError(f"Configuration section '{part}' is missing.")
                current = current[part]

            # Perform type validation based on existing value
            if parts[-1] not in current:
                raise ValueError(f"Configuration parameter '{parts[-1]}' is missing in section '{'.'.join(parts[:-1])}'.")

            existing_value = current[parts[-1]]
            if not isinstance(value, type(existing_value)):
                raise ValueError(
                    f"Invalid type for {path}. Expected {type(existing_value).__name__}, "
                    f"but got {type(value).__name__}."
                )

            # Update value
            current[parts[-1]] = value

            # Validate after update
            self.validate_config()

            # Save updated config
            self.save_config(self.config_path)

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise

    def save_config(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            path: Save path
        """
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise

    def get_env_value(self, name: str) -> Any:
        """
        Get value from environment variable.

        Args:
            name: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ValueError: If environment variable is not set
        """
        if name not in os.environ:
            raise ValueError(f"Environment variable '{name}' is not set.")
        return self._convert_type(os.environ[name])

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config['logging']

if __name__ == "__main__":
    # Example usage
    config = Config("configs/config.yaml")
    print("Model config:", config.get_model_config())
    print("Training config:", config.get_training_config())

