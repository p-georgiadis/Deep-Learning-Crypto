# src/preprocessing/pipeline.py
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
from .feature_engineering import FeatureEngineer


class Pipeline:
    """
    Data preprocessing pipeline for cryptocurrency price prediction.

    This class handles all data preprocessing steps including:
    - Data cleaning and validation
    - Feature engineering
    - Sequence creation
    - Train/val/test splitting
    - Feature scaling
    """

    def __init__(self, sequence_length=60, prediction_length=1, test_size=0.2, validation_size=0.2):
        """
        Initialize the preprocessing pipeline.

        Args:
            sequence_length: Length of input sequences
            prediction_length: Number of future time steps to predict
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaler = RobustScaler()
        self.feature_engineer = FeatureEngineer()
        total_split = test_size + validation_size
        if total_split >= 1.0:
            raise ValueError("The sum of test_size and validation_size must be less than 1.")

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

    def augment_data(self, X, y, noise_level=0.01, shift_range=2):
        """Add time series augmentation"""
        X_aug, y_aug = [], []

        # Original data
        X_aug.extend(X)
        y_aug.extend(y)

        # Add noise
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.extend(X + noise)
        y_aug.extend(y)

        # Time shift
        for shift in range(-shift_range, shift_range + 1):
            if shift != 0:
                X_aug.extend(np.roll(X, shift, axis=1))
                y_aug.extend(y)

        return np.array(X_aug), np.array(y_aug)

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Required: {required_columns}")

        if df.isnull().any().any():
            raise ValueError("DataFrame contains missing values")

    '''
    def process_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data using a wider IQR range for crypto."""
        df_clean = df.copy()

        for column in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 5 * IQR  # Increased from 3 to 5 for crypto
            upper_bound = Q3 + 5 * IQR

            df_clean[column] = df_clean[column].clip(lower=lower_bound,
                                                     upper=upper_bound)

        return df_clean
    '''

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the model.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional features
        """
        return self.feature_engineer.add_technical_features(df)

    def fit_normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler on the training data and transform it.

        Args:
            df: Training DataFrame to fit and transform.

        Returns:
            Normalized DataFrame.
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns
        self.numeric_features = numeric_features  # Save for later use
        scaled_data = self.scaler.fit_transform(df[numeric_features])
        return pd.DataFrame(scaled_data, columns=numeric_features, index=df.index)

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the already fitted scaler.

        Args:
            df: Input DataFrame to transform.

        Returns:
            Normalized DataFrame.
        """
        if not hasattr(self.scaler, 'center_'):
            raise ValueError("Scaler has not been fitted yet. Call 'fit_normalize_features' first.")
        numeric_features = self.numeric_features  # Use the same features as during fitting
        scaled_data = self.scaler.transform(df[numeric_features])
        return pd.DataFrame(scaled_data, columns=numeric_features, index=df.index)

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            df: Input DataFrame with scaled features and target.

        Returns:
            Tuple of (X sequences, y targets).
        """
        X, y = [], []
        for i in range(len(df) - self.sequence_length - self.prediction_length + 1):
            # Create a sequence of features
            sequence = df.iloc[i:(i + self.sequence_length)]
            # Create the target based on the forecast horizon
            target = df.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_length]['close']

            X.append(sequence.values)
            y.append(target.values)

        # Debugging: Ensure alignment of sequences and targets
        assert len(X) == len(y), f"Sequence and target length mismatch: {len(X)} != {len(y)}"
        self.logger.info(f"Created {len(X)} sequences with {self.sequence_length} time steps each.")
        return np.array(X), np.array(y)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame with features.

        Returns:
            Tuple of (train, val, test DataFrames).
        """
        train_size = 1 - self.test_size - self.validation_size

        n = len(df)
        train_end = int(n * train_size)
        val_end = train_end + int(n * self.validation_size)

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        # Debugging: Check if the splits are sufficient
        min_required_length = self.sequence_length + self.prediction_length - 1
        assert len(train) >= min_required_length, f"Insufficient training data for sequences (length: {len(train)})."
        assert len(val) >= min_required_length, f"Insufficient validation data for sequences (length: {len(val)})."
        assert len(test) >= min_required_length, f"Insufficient test data for sequences (length: {len(test)})."

        self.logger.info(f"Data split complete: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test

    def save_processed_data(self, processed_data: Dict[str, Union[np.ndarray, RobustScaler]], save_dir: str) -> None:
        """
        Save preprocessed data to disk.

        Args:
            processed_data: Dictionary containing processed data (X_train, y_train, etc.)
            save_dir: Directory to save the processed data
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                np.save(save_path / f"{key}.npy", value)
            elif isinstance(value, RobustScaler):
                # Save the scaler using joblib
                joblib.dump(value, save_path / f"{key}.joblib")
            else:
                self.logger.warning(
                    f"Unrecognized data type for key '{key}' and value type '{type(value)}'. Skipping save.")

    @staticmethod
    def load_processed_data(load_dir: str) -> Dict[str, Union[np.ndarray, RobustScaler]]:
        """
        Load preprocessed data from disk.

        Args:
            load_dir: Directory where processed data is stored

        Returns:
            Dictionary with processed data
        """
        load_path = Path(load_dir)
        processed_data = {}

        for file in load_path.iterdir():
            if file.suffix == '.npy':
                processed_data[file.stem] = np.load(file, allow_pickle=True)
            elif file.suffix == '.joblib':
                processed_data[file.stem] = joblib.load(file)
            else:
                # Skip other files or handle as needed
                continue

        return processed_data

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocess new data for prediction.

        Args:
            df: Raw input DataFrame for prediction.

        Returns:
            Dictionary containing preprocessed sequences ('X') and other metadata if needed.
        """
        try:
            self.logger.info("Starting preprocessing for prediction data")

            # Step 1: Validate input data
            self.validate_data(df)

            # Step 2: Handle outliers
            df_clean = self.process_outliers(df)

            # Step 3: Create features
            df_features = self.create_features(df_clean)

            # Step 4: Handle missing values
            if df_features.isnull().any().any():
                self.logger.warning("Missing values detected after feature engineering. Filling or dropping NaNs.")
                df_features.fillna(method='bfill', inplace=True)  # Backfill
                df_features.dropna(inplace=True)  # Drop remaining NaNs

            # Step 5: Normalize features
            if not hasattr(self.scaler, 'center_') or not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before transforming data.")
            normalized_data = self.scaler.transform(df_features)
            df_normalized = pd.DataFrame(
                normalized_data,
                columns=df_features.columns,
                index=df_features.index
            )

            # Step 6: Create sequences
            self.logger.info("Creating sequences for prediction")
            X, _ = self.prepare_sequences(df_normalized)

            self.logger.info(f"Prediction data preprocessing complete. Sequences shape: {X.shape}")
            return {'X': X}

        except Exception as e:
            self.logger.error(f"Error preprocessing prediction data: {str(e)}")
            raise

    def run(self, df: pd.DataFrame, save_dir: str = None, prediction_mode: bool = False) -> Dict[
        str, Union[np.ndarray, RobustScaler]]:
        """
        Run the complete preprocessing pipeline.

        Args:
            df: Raw input DataFrame.
            save_dir: Directory to save processed data (optional).
            prediction_mode: Whether to process data for prediction.

        Returns:
            Dictionary containing processed data or scaler (for prediction mode).
        """
        try:
            self.logger.info("Starting preprocessing pipeline")

            # Step 1: Validate input data
            self.validate_data(df)

            # Step 2: Process outliers
            #df_clean = self.process_outliers(df)

            # Step 3: Create features
            df_features = self.create_features(df)
            self.logger.info(f"Data length after feature engineering: {len(df_features)}")

            # **Step 4: Handle missing values**
            if df_features.isnull().any().any():
                self.logger.warning("Missing values detected after feature engineering")
                df_features = df_features.bfill()
                df_features.dropna(inplace=True)

            # **Step 5: Split data into train, validation, and test sets**
            # Now split the data
            train_df, val_df, test_df = self.split_data(df_features)

            # **Step 6: Normalize features using only training data**
            train_normalized = self.fit_normalize_features(train_df)

            # Use the same columns as the training set for validation and test
            val_normalized = self.normalize_features(val_df)
            test_normalized = self.normalize_features(test_df)

            # Debugging: Verify feature consistency
            self.logger.info(f"Train features: {train_normalized.columns.tolist()}")
            self.logger.info(f"Validation features: {val_normalized.columns.tolist()}")
            self.logger.info(f"Test features: {test_normalized.columns.tolist()}")

            # **Step 7: Create sequences**
            X_train, y_train = self.prepare_sequences(train_normalized)
            X_val, y_val = self.prepare_sequences(val_normalized)
            X_test, y_test = self.prepare_sequences(test_normalized)

            # Debugging: Check example sequences and targets
            for i in range(min(5, len(X_train))):
                self.logger.info(f"X_train[{i}] = {X_train[i]}")
                self.logger.info(f"y_train[{i}] = {y_train[i]}")

            self.logger.info(f"Preprocessing complete. Training sequences shape: {X_train.shape}")

            processed_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': self.scaler,
                'numeric_features': self.numeric_features
            }

            # Save the processed data if a directory is provided
            if save_dir:
                self.save_processed_data(processed_data, save_dir)

            return processed_data

        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

    def load_scaler(self, scaler_path: str):
        """
        Load the scaler from a file.

        Args:
            scaler_path: Path to the saved scaler file.
        """
        self.scaler = joblib.load(scaler_path)