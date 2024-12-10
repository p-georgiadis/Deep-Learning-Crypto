# src/preprocessing/pipeline.py

import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import json

from .feature_engineering import FeatureEngineer
from src.utils.config import Config

class Pipeline:
    """
    The Pipeline class handles:
    - Data validation and cleaning
    - Feature engineering
    - Normalization/scaling of non-target features and the target (close) separately
    - For training: splitting data and preparing sequences (X, Y).
    - For prediction: returning only the last sequence_length rows scaled for input.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        sequence_length: int = 60,
        prediction_length: int = 1,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        feature_scaler_type: str = 'standard',
        target_scaler_type: str = 'robust',
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.config = config

        if config and isinstance(config, Config):
            model_config = config.get_model_config()
            data_config = config.get_data_config()
            preprocessing_config = config.get_preprocessing_config()

            self.sequence_length = model_config.get('sequence_length', sequence_length)
            self.prediction_length = model_config.get('prediction_length', prediction_length)

            self.test_size = data_config.get('test_split', test_size)
            self.validation_size = data_config.get('validation_split', validation_size)

            scaling_config = preprocessing_config.get('scaling', {})
            self.feature_scaler_type = scaling_config.get('feature_scaler_type', feature_scaler_type)
            self.target_scaler_type = scaling_config.get('target_scaler_type', target_scaler_type)
        else:
            self.sequence_length = sequence_length
            self.prediction_length = prediction_length
            self.test_size = test_size
            self.validation_size = validation_size
            self.feature_scaler_type = feature_scaler_type
            self.target_scaler_type = target_scaler_type

        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("Sum of test_size and validation_size must be less than 1.")

        self.scaler = self._initialize_scaler(self.feature_scaler_type)
        self.target_scaler = self._initialize_scaler(self.target_scaler_type)

        self.numeric_features = []
        self.scalers_fitted = False

        fe_config = None
        if self.config:
            preprocessing_config = self.config.get_preprocessing_config()
            if preprocessing_config and 'feature_engineering' in preprocessing_config:
                fe_config = preprocessing_config['feature_engineering']

        self.feature_engineer = FeatureEngineer(config=fe_config)

        self.logger.info(f"Pipeline initialized with sequence_length={self.sequence_length}, "
                         f"prediction_length={self.prediction_length}, test_size={self.test_size}, "
                         f"validation_size={self.validation_size}, "
                         f"feature_scaler_type={self.feature_scaler_type}, target_scaler_type={self.target_scaler_type}")

    @staticmethod
    def _initialize_scaler(scaler_type: str):
        if scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def _update_numeric_features(self, df: pd.DataFrame):
        # Exclude 'close' from numeric_features to ensure it's only scaled by target_scaler
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features = [col for col in numeric_cols if col != 'close']
        self.logger.info(f"Numeric features identified (excluding close): {self.numeric_features}")

    def validate_data(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df[required_columns].isnull().values.any():
            raise ValueError("Required columns contain missing values.")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Applying feature engineering...")
        df_features = self.feature_engineer.add_technical_features(df)
        return df_features

    def fit_normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        df_scaled = df.copy()
        self._update_numeric_features(df_scaled)

        if self.config:
            scaling_config = self.config.get_preprocessing_config().get('scaling', {})
            if scaling_config.get('price_transform') == 'log1p':
                self.logger.info("Applying log1p transformation to price columns (open, high, low, close).")
                price_cols = ['open', 'high', 'low', 'close']
                df_scaled[price_cols] = np.log1p(df_scaled[price_cols])

            if scaling_config.get('volume_transform') == 'log1p':
                self.logger.info("Applying log1p transformation to volume.")
                df_scaled['volume'] = np.log1p(df_scaled['volume'])

        # Scale numeric features (excluding 'close')
        if self.numeric_features:
            self.logger.info("Fitting scaler on numeric features.")
            df_scaled[self.numeric_features] = self.scaler.fit_transform(df_scaled[self.numeric_features])

        # Scale target close separately
        if 'close' in df_scaled.columns:
            self.logger.info("Fitting scaler on target ('close').")
            df_scaled['close'] = self.target_scaler.fit_transform(df_scaled[['close']]).flatten()

        self.scalers_fitted = True
        return df_scaled

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if not self.scalers_fitted:
            raise ValueError("Scalers have not been fitted. Call fit_normalize_features first.")

        df_scaled = df.copy()

        if self.config:
            scaling_config = self.config.get_preprocessing_config().get('scaling', {})
            if scaling_config.get('price_transform') == 'log1p':
                self.logger.info("Applying log1p transform to price columns.")
                price_cols = ['open', 'high', 'low', 'close']
                existing_price_cols = [c for c in price_cols if c in df_scaled.columns]
                if existing_price_cols:
                    df_scaled[existing_price_cols] = np.log1p(df_scaled[existing_price_cols])

            if scaling_config.get('volume_transform') == 'log1p' and 'volume' in df_scaled.columns:
                self.logger.info("Applying log1p transform to volume.")
                df_scaled['volume'] = np.log1p(df_scaled['volume'])

        # Temporarily remove close for feature scaling
        close_col = None
        if 'close' in df_scaled.columns:
            close_col = df_scaled['close'].copy()
            df_scaled.drop(columns=['close'], inplace=True, errors='ignore')

        # Ensure only known numeric_features remain
        all_current_numeric = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        current_set = set(all_current_numeric)
        expected_set = set(self.numeric_features)

        extra_features = current_set - expected_set
        if extra_features:
            self.logger.warning(f"Dropping extra numeric features not seen at fit time: {extra_features}")
            df_scaled.drop(columns=list(extra_features), inplace=True, errors='ignore')

        missing_features = expected_set - set(df_scaled.columns)
        for mf in missing_features:
            self.logger.warning(f"Missing feature {mf} at prediction time. Filling with zeros.")
            df_scaled[mf] = 0.0

        df_scaled = df_scaled.reindex(columns=self.numeric_features, fill_value=0.0)
        df_scaled[self.numeric_features] = self.scaler.transform(df_scaled[self.numeric_features])

        # Re-add close using target_scaler if it existed
        if close_col is not None:
            close_scaled = self.target_scaler.transform(close_col.to_frame()).flatten()
            df_scaled['close'] = close_scaled

        return df_scaled

    def run(self, df: pd.DataFrame, save_dir: Optional[str] = None, prediction_mode: bool = False) -> Dict[str, np.ndarray]:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        self.validate_data(df)
        df_features = self.create_features(df)

        if df_features.isnull().values.any():
            self.logger.warning("Missing values detected, applying bfill and ffill.")
            df_features.bfill(inplace=True)
            df_features.ffill(inplace=True)

        if df_features.empty:
            raise ValueError("No data left after handling missing values.")

        self._update_numeric_features(df_features)

        if prediction_mode:
            if self.scalers_fitted:
                df_normalized = self.normalize_features(df_features)
            else:
                df_normalized = self.fit_normalize_features(df_features)

            if len(df_normalized) < self.sequence_length:
                raise ValueError("Not enough data to form one sequence for prediction.")

            last_seq = df_normalized.iloc[-self.sequence_length:]
            X = np.expand_dims(last_seq.values, axis=0)
            return {'X': X}

        train_df, val_df, test_df = self.split_data(df_features)

        train_norm = self.fit_normalize_features(train_df)
        val_norm = self.normalize_features(val_df)
        test_norm = self.normalize_features(test_df)

        X_train, Y_train = self.prepare_sequences(train_norm)
        X_val, Y_val = self.prepare_sequences(val_norm)
        X_test, Y_test = self.prepare_sequences(test_norm)

        result = {
            'X_train': X_train,
            'y_train': Y_train,
            'X_val': X_val,
            'y_val': Y_val,
            'X_test': X_test,
            'y_test': Y_test,
            'numeric_features': self.numeric_features
        }

        if save_dir:
            self.save_processed_data(result, save_dir)
            scaler_dir = Path(save_dir) / "scalers"
            scaler_dir.mkdir(parents=True, exist_ok=True)
            self.save_scaler(
                scaler_dir / "feature_scaler.joblib",
                scaler_dir / "target_scaler.joblib"
            )

        return result

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if df.empty:
            raise ValueError("Empty DataFrame, cannot prepare sequences.")
        if len(df) < self.sequence_length + self.prediction_length:
            raise ValueError("Not enough data to create sequences.")

        X, Y = [], []
        max_start_idx = len(df) - self.sequence_length - self.prediction_length + 1

        for i in range(max_start_idx):
            seq = df.iloc[i:i + self.sequence_length]
            tgt_idx = i + self.sequence_length
            tgt_close = df.iloc[tgt_idx]['close']

            prev_close = df.iloc[tgt_idx - 1]['close'] if (tgt_idx - 1) >= 0 else tgt_close

            Y.append([tgt_close, prev_close])
            X.append(seq.values)

        self.logger.info(f"Created {len(X)} sequences of length {self.sequence_length}.")
        return np.array(X), np.array(Y)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        min_size = max(self.sequence_length + self.prediction_length, 200)
        if len(df) < min_size * 3:
            raise ValueError("Not enough data for splitting into train/val/test.")

        train_size = 1 - self.test_size - self.validation_size
        n = len(df)
        buffer_size = 100
        train_end = int(n * train_size)
        val_end = train_end + int(n * self.validation_size)

        train = df.iloc[:train_end + buffer_size].copy()
        val = df.iloc[max(0, train_end - buffer_size):val_end + buffer_size].copy()
        test = df.iloc[max(0, val_end - buffer_size):].copy()

        self.logger.info(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test

    def save_processed_data(self, processed_data: Dict[str, np.ndarray], save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                np.save(save_path / f"{key}.npy", value)
            elif key == 'numeric_features' and isinstance(value, list):
                with open(save_path / f"{key}.json", 'w') as f:
                    json.dump(value, f)
            else:
                self.logger.warning(f"Skipping saving {key}, unrecognized type {type(value)}")

    def save_scaler(self, scaler_path: Union[str, Path], target_scaler_path: Union[str, Path]):
        scaler_path = Path(scaler_path)
        target_scaler_path = Path(target_scaler_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.target_scaler, target_scaler_path)
        self.logger.info(f"Saved scalers to {scaler_path} and {target_scaler_path}")
        self.scalers_fitted = True

    def load_scaler(self, scaler_path: Union[str, Path], target_scaler_path: Union[str, Path]):
        scaler_path = Path(scaler_path)
        target_scaler_path = Path(target_scaler_path)

        if not scaler_path.exists() or not target_scaler_path.exists():
            raise FileNotFoundError("Scaler files not found.")

        self.scaler = joblib.load(scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        self.scalers_fitted = True
        self.logger.info(f"Loaded scalers from {scaler_path} and {target_scaler_path}")

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        inverse_scaled = self.target_scaler.inverse_transform(predictions)
        if self.config and self.config.get_preprocessing_config().get('scaling', {}).get('price_transform') == 'log1p':
            inverse_scaled = np.expm1(inverse_scaled)

        return inverse_scaled.flatten()

    def inverse_transform_actuals(self, y: np.ndarray) -> np.ndarray:
        actual_prices = y[:,0].reshape(-1,1)
        inverse_actual = self.target_scaler.inverse_transform(actual_prices)
        if self.config and self.config.get_preprocessing_config().get('scaling', {}).get('price_transform') == 'log1p':
            inverse_actual = np.expm1(inverse_actual)
        return inverse_actual.flatten()

    def save(self, path: Union[str, Path]):
        path = Path(path)
        joblib.dump({
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'numeric_features': self.numeric_features
        }, path)
        self.logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]):
        data = joblib.load(path)
        pipeline = cls()
        pipeline.scaler = data['scaler']
        pipeline.target_scaler = data['target_scaler']
        pipeline.numeric_features = data['numeric_features']
        pipeline.scalers_fitted = True
        logging.info(f"Pipeline loaded from {path}")
        return pipeline
