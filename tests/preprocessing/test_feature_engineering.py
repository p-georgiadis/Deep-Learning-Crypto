import pytest
import pandas as pd
import numpy as np
from src.preprocessing.feature_engineering import FeatureEngineer


@pytest.fixture
def engineer():
    """Fixture to create a shared FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'open': np.random.uniform(30000, 40000, 100),
        'high': np.random.uniform(35000, 45000, 100),
        'low': np.random.uniform(25000, 35000, 100),
        'close': np.random.uniform(30000, 40000, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    }, index=dates)


def test_calculate_rsi(engineer):
    """Test RSI calculation."""
    dates = pd.date_range(start='2021-01-01', periods=20, freq='D')
    prices = pd.Series([
        100, 102, 104, 103, 106, 105, 107, 108, 109, 110,
        108, 106, 104, 102, 100, 98, 96, 94, 92, 90
    ], index=dates)

    rsi = engineer.calculate_rsi(prices, periods=14)

    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(prices)
    assert all(0 <= x <= 100 for x in rsi.dropna())
    assert rsi.iloc[14] >= 50  # Adjusted to allow for RSI = 50
    assert rsi.iloc[-1] < 50  # Expect downtrend RSI < 50


def test_calculate_macd(engineer):
    """Test MACD calculation."""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    prices = pd.Series(np.linspace(100, 200, 100), index=dates)  # Upward trend

    macd, signal, hist = engineer.calculate_macd(prices)

    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert len(macd) == len(prices)
    assert len(signal) == len(prices)
    assert len(hist) == len(prices)
    assert macd.iloc[-1] > 0  # MACD should be positive in an uptrend


def test_calculate_bollinger_bands(engineer):
    """Test Bollinger Bands calculation."""
    dates = pd.date_range(start='2021-01-01', periods=50, freq='D')
    prices = pd.Series(np.random.normal(100, 10, 50), index=dates)

    upper, middle, lower = engineer.calculate_bollinger_bands(prices, window=20)

    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(prices)

    # Drop NaN values before comparison to ensure valid checks
    valid_upper = upper.dropna()
    valid_middle = middle.dropna()
    valid_lower = lower.dropna()

    assert all(valid_upper > valid_middle)
    assert all(valid_middle > valid_lower)



def test_add_all_features(engineer, sample_price_data):
    """Test adding all technical features."""
    df_with_features = engineer.add_technical_features(sample_price_data)

    expected_features = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'sma_20', 'sma_50',
        'daily_return', 'volatility', 'roc', 'momentum'
    ]

    assert all(feature in df_with_features.columns for feature in expected_features)
    assert not df_with_features.isnull().any().any()


def test_calculate_moving_averages(engineer):
    """Test moving averages calculation."""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    prices = pd.Series(np.linspace(100, 200, 100), index=dates)  # Linear upward trend

    sma_20, sma_50 = engineer.calculate_moving_averages(prices)

    assert isinstance(sma_20, pd.Series)
    assert isinstance(sma_50, pd.Series)
    assert len(sma_20) == len(prices)
    assert len(sma_50) == len(prices)
    assert sma_20.iloc[-1] > sma_50.iloc[-1]  # Shorter MA should be higher in an uptrend


def test_calculate_volatility(engineer):
    """Test volatility calculation."""
    dates = pd.date_range(start='2021-01-01', periods=50, freq='D')
    prices = pd.Series(
        np.concatenate([
            np.random.normal(100, 1, 25),  # Low volatility
            np.random.normal(100, 10, 25)  # High volatility
        ]),
        index=dates
    )

    volatility = engineer.calculate_volatility(prices, window=10)

    assert isinstance(volatility, pd.Series)
    assert len(volatility) == len(prices)
    assert volatility.iloc[-1] > volatility.iloc[15]  # Volatility should be higher in the second half


def test_feature_engineering_with_missing_data(engineer):
    """Test feature engineering with missing data."""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(30000, 40000, 100),
        'high': np.random.uniform(35000, 45000, 100),
        'low': np.random.uniform(25000, 35000, 100),
        'close': np.random.uniform(30000, 40000, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    }, index=dates)

    # Introduce missing values
    data.loc[data.index[0:5], 'close'] = np.nan

    # Test that a ValueError is raised for missing critical column data
    with pytest.raises(ValueError, match="Missing values detected in critical columns"):
        engineer.add_technical_features(data)


def test_calculate_momentum_indicators(engineer):
    """Test momentum indicators calculation."""
    dates = pd.date_range(start='2021-01-01', periods=50, freq='D')
    prices = pd.Series(np.random.normal(100, 10, 50), index=dates)

    roc, mom = engineer.calculate_momentum_indicators(prices)

    assert isinstance(roc, pd.Series)
    assert isinstance(mom, pd.Series)
    assert len(roc) == len(prices)
    assert len(mom) == len(prices)
    assert not roc.iloc[12:].isnull().any()  # Skip initial NaN values
    assert not mom.iloc[12:].isnull().any()  # Skip initial NaN values
