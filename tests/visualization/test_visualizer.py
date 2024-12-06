import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


def test_plot_feature_importance():
    """Test feature importance plotting."""
    from src.visualization.visualizer import CryptoVisualizer

    features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    importance_scores = [0.3, 0.2, 0.15, 0.2, 0.15]

    visualizer = CryptoVisualizer()
    fig = visualizer.plot_feature_importance(features, importance_scores)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Should contain bar plot
    assert fig.data[0].type == 'bar'
    assert list(fig.data[0].x) == features
    assert list(fig.data[0].y) == importance_scores


def test_save_plot(tmp_path, sample_price_history):
    """Test saving a plot to a valid file."""
    from src.visualization.visualizer import CryptoVisualizer

    visualizer = CryptoVisualizer()
    fig = visualizer.plot_price_history(sample_price_history)

    save_path = tmp_path / "test_plot.html"
    visualizer.save_plot(fig, save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_prediction_intervals_with_confidence(sample_price_history):
    """Test prediction intervals with confidence intervals."""
    from src.visualization.visualizer import CryptoVisualizer

    mean_predictions = np.random.uniform(35000, 45000, 30)
    lower_bound = mean_predictions - 500
    upper_bound = mean_predictions + 500

    visualizer = CryptoVisualizer()
    fig = visualizer.plot_prediction_intervals(
        actual_prices=sample_price_history['close'][-30:],
        predicted_means=mean_predictions,
        lower_bounds=lower_bound,
        upper_bounds=upper_bound
    )

    assert isinstance(fig, go.Figure)
    assert any(trace.name == "Prediction Interval" for trace in fig.data)


def test_plot_interactive_dashboard(sample_price_history, sample_predictions, sample_training_history):
    """Test interactive dashboard creation."""
    from src.visualization.visualizer import CryptoVisualizer

    visualizer = CryptoVisualizer()
    dashboard = visualizer.create_interactive_dashboard(
        price_history=sample_price_history,
        predictions=sample_predictions,
        training_history=sample_training_history
    )

    assert isinstance(dashboard, go.Figure)
    assert len(dashboard.layout.annotations) >= 4  # Expect at least 4 annotations


def test_plot_performance_metrics():
    """Test performance metrics visualization."""
    from src.visualization.visualizer import CryptoVisualizer

    metrics = {
        'MAE': 100.5,
        'RMSE': 150.3,
        'MAPE': 2.5,
        'Directional Accuracy': 65.0
    }

    visualizer = CryptoVisualizer()
    fig = visualizer.plot_performance_metrics(metrics)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Should contain bar plot
    assert fig.data[0].type == 'bar'
    assert list(fig.data[0].x) == list(metrics.keys())
    assert list(fig.data[0].y) == list(metrics.values())


def test_invalid_input_handling():
    """Test handling of invalid inputs."""
    from src.visualization.visualizer import CryptoVisualizer

    visualizer = CryptoVisualizer()

    # Test with empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame must contain OHLCV data"):
        visualizer.plot_price_history(pd.DataFrame())

    # Test with missing OHLC columns
    invalid_df = pd.DataFrame({'price': [1, 2, 3]})
    with pytest.raises(KeyError, match="'open'"):  # Match the actual missing column
        visualizer.plot_technical_indicators(invalid_df)

    # Test with mismatched prediction lengths
    with pytest.raises(ValueError, match="Lengths of actual and predicted prices must match"):
        visualizer.plot_predictions(
            actual_prices=pd.Series([1, 2, 3]),
            predicted_prices=pd.Series([1, 2])
        )


def test_plot_correlation_matrix(sample_technical_data):
    """Test correlation matrix plotting."""
    from src.visualization.visualizer import CryptoVisualizer

    visualizer = CryptoVisualizer()
    fig = visualizer.plot_correlation_matrix(sample_technical_data)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Should contain heatmap
    assert fig.data[0].type == 'heatmap'


# Utility function to ensure required fixtures are available
@pytest.fixture
def sample_price_history():
    """Generate a sample price history DataFrame."""
    dates = pd.date_range(start="2021-01-01", periods=100, freq="D")
    data = {
        "open": np.random.uniform(30000, 40000, len(dates)),
        "high": np.random.uniform(40000, 45000, len(dates)),
        "low": np.random.uniform(25000, 30000, len(dates)),
        "close": np.random.uniform(30000, 40000, len(dates)),
        "volume": np.random.uniform(1e5, 1e6, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_training_history():
    """Generate a sample training history dictionary."""
    return {
        "loss": [0.05, 0.04, 0.035, 0.032, 0.03],
        "mae": [0.04, 0.035, 0.03, 0.028, 0.027],
        "val_loss": [0.055, 0.045, 0.04, 0.038, 0.037],
        "val_mae": [0.045, 0.04, 0.035, 0.033, 0.032]
    }


@pytest.fixture
def sample_technical_data(sample_price_history):
    """Generate sample technical data."""
    df = sample_price_history.copy()
    df["rsi"] = np.random.uniform(30, 70, len(df))
    df["macd"] = np.random.uniform(-10, 10, len(df))
    df["macd_signal"] = np.random.uniform(-10, 10, len(df))
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df.dropna(inplace=True)
    return df
