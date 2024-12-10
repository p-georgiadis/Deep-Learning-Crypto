# src/utils/config.py

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml

class Config:
    """
    A simpler configuration manager:
    - Loads a YAML config file.
    - Provides getters for common sections (paths, model, training, etc.).
    - Ensures directories exist.
    - Basic validation (e.g., required keys) can be done, but kept minimal.
    """

    def __init__(self, config_path: Union[str, Path]):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a top-level dictionary.")

        self.logger.info(f"Configuration loaded from {self.config_path}")
        return config

    def _create_directories(self) -> None:
        """
        Create all directories specified in the configuration under 'paths'.
        """
        paths = self.config.get('paths', {})
        for key, path_str in paths.items():
            p = Path(path_str)
            p.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory for {key}: {p}")

    def get_path(self, key: str) -> str:
        """
        Retrieve a path from config['paths'] by key.
        Raises KeyError if not found.
        """
        paths = self.config.get('paths', {})
        if key in paths:
            return paths[key]
        raise KeyError(f"Path '{key}' not found in config.")

    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})

    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    def get_preprocessing_config(self) -> Dict[str, Any]:
        return self.config.get('preprocessing', {})

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})

    def get_model_dir(self) -> str:
        return self.get_path('models_dir')
