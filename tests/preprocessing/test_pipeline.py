# tests/preprocessing/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.pipeline import Pipeline


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing"""
    sequence_length = 60
    prediction_length = 1
    total_rows = sequence_length + prediction_length + 100  # Add buffer rows
    dates = pd.date_range(start='2021-01-01', periods=total_rows, freq='D')

    return pd.DataFrame({
        'open': np.random.uniform(30000, 40000, total_rows),
        'high': np.random.uniform(35000, 45000, total_rows),
        'low': np.random.uniform(25000, 35000, total_rows),
        'close': np.random.uniform(30000, 40000, total_rows),
        'volume': np.random.uniform(1000000, 2000000, total_rows)
    }, index=dates)


@pytest.fixture
def sample_technical_data(sample_raw_data):
    """Create sample data with technical indicators"""
    from src.preprocessing.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    return engineer.add_technical_features(sample_raw_data)


def test_pipeline_initialization():
    """Test Pipeline initialization"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline(sequence_length=60, prediction_length=30)
    assert pipeline.sequence_length == 60
    assert pipeline.prediction_length == 30
    assert hasattr(pipeline, 'scaler')


def test_prepare_sequences(sample_technical_data):
    """Test sequence preparation"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline(sequence_length=5, prediction_length=1)
    X, y = pipeline.prepare_sequences(sample_technical_data)

    # Check shapes
    assert len(X.shape) == 3  # (samples, sequence_length, features)
    assert len(y.shape) == 2  # (samples, prediction_length)
    assert X.shape[1] == 5  # sequence_length
    assert y.shape[1] == 1  # prediction_length


def test_normalize_features(sample_technical_data):
    """Test feature normalization"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline()
    normalized_df = pipeline.normalize_features(sample_technical_data)

    # Check if values are normalized with a small tolerance
    assert normalized_df.min().min() >= -1 - 1e-9
    assert normalized_df.max().max() <= 1 + 1e-9
    assert isinstance(normalized_df, pd.DataFrame)
    assert normalized_df.shape == sample_technical_data.shape


def test_create_features(sample_technical_data):
    """Test feature creation"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline()
    features_df = pipeline.create_features(sample_technical_data)

    expected_features = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'sma_20', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower',
        'volatility', 'daily_return', 'roc', 'momentum'
    ]

    assert all(feature in features_df.columns for feature in expected_features)
    assert not features_df.isnull().any().any()


def test_split_data(sample_technical_data):
    """Test data splitting"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline()
    train, val, test = pipeline.split_data(sample_technical_data)

    total_len = len(sample_technical_data)
    assert len(train) + len(val) + len(test) == total_len
    assert len(train) > len(val)
    assert abs(len(val) - len(test)) <= 1  # Allow for a difference of 1 due to rounding


def test_process_outliers(sample_technical_data):
    """Test outlier processing"""
    from src.preprocessing.pipeline import Pipeline

    # Add some outliers
    sample_technical_data.loc[sample_technical_data.index[0], 'close'] = 1000000

    pipeline = Pipeline()
    cleaned_df = pipeline.process_outliers(sample_technical_data)

    assert cleaned_df['close'].max() < 1000000
    assert not cleaned_df.isnull().any().any()


def test_complete_pipeline(sample_raw_data):
    """Test complete pipeline process"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline(sequence_length=60, prediction_length=1)
    result = pipeline.run(sample_raw_data)

    assert 'X_train' in result
    assert 'y_train' in result
    assert 'X_val' in result
    assert 'y_val' in result
    assert 'X_test' in result
    assert 'y_test' in result
    assert 'scaler' in result

    # Ensure sequences are generated
    assert result['X_train'].size > 0, "X_train is empty"
    assert result['y_train'].size > 0, "y_train is empty"


def test_pipeline_with_missing_data():
    """Test pipeline handling of missing data"""
    from src.preprocessing.pipeline import Pipeline

    # Create data with missing values
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(30000, 40000, 100),
        'high': np.random.uniform(35000, 45000, 100),
        'low': np.random.uniform(25000, 35000, 100),
        'close': np.random.uniform(30000, 40000, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    }, index=dates)

    # Insert some NaN values
    data.loc[data.index[0:5], 'close'] = np.nan

    pipeline = Pipeline()
    with pytest.raises(ValueError):
        pipeline.run(data)

def test_save_processed_data(tmp_path):
    pipeline = Pipeline(sequence_length=60, prediction_length=1)
    # Generate a dummy dataset and fit the scaler
    dummy_data = np.random.rand(100, 5)
    pipeline.scaler.fit(dummy_data)

    processed_data = {
        'X_train': np.random.rand(100, 60, 5),
        'y_train': np.random.rand(100, 1),
        'scaler': pipeline.scaler  # Fitted scaler
    }
    pipeline.save_processed_data(processed_data, tmp_path)

    assert (tmp_path / "X_train.npy").exists()
    assert (tmp_path / "y_train.npy").exists()
    assert (tmp_path / "scaler.json").exists()


def test_load_processed_data(tmp_path):
    pipeline = Pipeline(sequence_length=60, prediction_length=1)
    # Generate a dummy dataset and fit the scaler
    dummy_data = np.random.rand(100, 5)
    pipeline.scaler.fit(dummy_data)

    processed_data = {
        'X_train': np.random.rand(100, 60, 5),
        'y_train': np.random.rand(100, 1),
        'scaler': pipeline.scaler  # Fitted scaler
    }
    pipeline.save_processed_data(processed_data, tmp_path)

    loaded_data = pipeline.load_processed_data(tmp_path)
    np.testing.assert_array_equal(processed_data['X_train'], loaded_data['X_train'])
    assert isinstance(loaded_data['scaler'], MinMaxScaler)


def test_complete_pipeline_with_save(sample_raw_data, tmp_path):
    """Test complete pipeline process with saving preprocessed data."""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline(sequence_length=60, prediction_length=1)
    result = pipeline.run(sample_raw_data, save_dir=str(tmp_path))

    # Ensure data was saved
    assert (tmp_path / "X_train.npy").exists()
    assert (tmp_path / "y_train.npy").exists()
    assert (tmp_path / "scaler.json").exists()

    # Validate loaded data matches the result
    loaded_data = pipeline.load_processed_data(tmp_path)
    np.testing.assert_array_equal(result['X_train'], loaded_data['X_train'])
    np.testing.assert_array_equal(result['y_train'], loaded_data['y_train'])

def test_pipeline_data_validation():
    """Test input data validation"""
    from src.preprocessing.pipeline import Pipeline

    pipeline = Pipeline()

    # Test with empty DataFrame
    with pytest.raises(ValueError):
        pipeline.run(pd.DataFrame())

    # Test with missing required columns
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    invalid_data = pd.DataFrame({
        'some_column': np.random.uniform(30000, 40000, 100)
    }, index=dates)

    with pytest.raises(ValueError):
        pipeline.run(invalid_data)
