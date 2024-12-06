import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from src.monitoring.dashboard import MonitoringDashboard


@pytest.fixture
def mock_streamlit():
    """Mock the Streamlit library."""
    with patch("src.monitoring.dashboard.st") as mock_st:
        yield mock_st


def test_load_config(mock_streamlit):
    """Test configuration loading."""
    mock_config = """
    paths:
        results_dir: "results"
    """

    with patch("builtins.open", new_callable=MagicMock) as mock_open, patch("yaml.safe_load", return_value={
        "paths": {"results_dir": "results"}}):
        mock_open.return_value.__enter__.return_value.read.return_value = mock_config
        dashboard = MonitoringDashboard()

        # Validate that config is loaded correctly
        assert "paths" in dashboard.config
        assert "results_dir" in dashboard.config["paths"]
        assert dashboard.config["paths"]["results_dir"] == "results"


def test_load_results(mock_streamlit, tmp_path):
    """Test loading results."""
    mock_config = {"paths": {"results_dir": str(tmp_path)}}

    with patch.object(MonitoringDashboard, "load_config", return_value=mock_config):
        dashboard = MonitoringDashboard()
        dashboard.config = mock_config

        results_dir = Path(tmp_path)
        results_dir.mkdir(parents=True, exist_ok=True)

        mock_result_path = results_dir / "btc_results_1.json"
        mock_result_path.write_text(json.dumps({
            "evaluation": {"mae": 0.01, "rmse": 0.02, "directional_accuracy": 0.9},
            "predictions": [30000, 31000],
            "actual_prices": [30000, 31500],
            "history": {
                "loss": [0.1, 0.05, 0.03],
                "val_loss": [0.2, 0.1, 0.05]
            }
        }))

        results = dashboard.load_results()

    assert "btc" in results
    assert results["btc"]["evaluation"]["mae"] == 0.01
    assert results["btc"]["history"]["loss"] == [0.1, 0.05, 0.03]


def test_show_system_metrics(mock_streamlit):
    """Test system metrics display."""
    dashboard = MonitoringDashboard()
    dashboard.show_system_metrics()

    assert mock_streamlit.metric.call_count == 4  # Verify all metrics are displayed


def test_live_predictions(mock_streamlit):
    """Test live predictions display."""
    dashboard = MonitoringDashboard()

    mock_results = {
        "btc": {
            "predictions": [31000.0],
            "actual_prices": [30500.0],
        }
    }

    # Updated method name to match the class definition
    dashboard.show_live_predictions(mock_results)
    assert mock_streamlit.metric.called


def test_plot_model_performance(mock_streamlit):
    """Test model performance plotting."""
    dashboard = MonitoringDashboard()

    mock_results = {
        "btc": {
            "evaluation": {"mae": 0.01, "rmse": 0.02, "directional_accuracy": 0.5},
            "predictions": [30000, 31000, 32000],
            "actual_prices": [30000, 30500, 31500],
            "history": {  # Add history key
                "loss": [0.1, 0.05, 0.03],
                "val_loss": [0.2, 0.1, 0.05]
            }
        }
    }

    dashboard.show_model_performance(mock_results)
    assert mock_streamlit.plotly_chart.called  # Verify chart is displayed




def test_training_history(mock_streamlit):
    """Test training history plotting."""
    dashboard = MonitoringDashboard()

    mock_results = {
        "btc": {
            "history": {
                "loss": [0.1, 0.05, 0.03],
                "val_loss": [0.2, 0.1, 0.05]
            }
        }
    }

    dashboard.show_training_history(mock_results)
    assert mock_streamlit.plotly_chart.called  # Verify the training history is plotted
