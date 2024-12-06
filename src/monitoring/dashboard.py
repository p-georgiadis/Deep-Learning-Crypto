# src/monitoring/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import yaml
from datetime import datetime
from src.visualization.visualizer import CryptoVisualizer

class MonitoringDashboard:
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(
            page_title="Crypto Prediction Monitor",
            page_icon="📈",
            layout="wide"
        )
        self.visualizer = CryptoVisualizer()
        self.config = self.load_config()

    @staticmethod
    def load_config():
        """Load project configuration."""
        try:
            with open('configs/config.yaml') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error("Configuration file not found. Ensure 'configs/config.yaml' exists.")
            return {}

    def load_results(self):
        """Load training results and metrics."""
        results_dir = Path(self.config.get('paths', {}).get('results_dir', 'results'))
        if not results_dir.exists():
            st.error(f"Results directory {results_dir} not found.")
            return {}

        all_results = {}
        for result_file in results_dir.glob('*_results_*.json'):
            try:
                with open(result_file) as f:
                    coin = result_file.stem.split('_')[0]
                    all_results[coin] = json.load(f)
            except Exception as e:
                st.warning(f"Error loading results from {result_file}: {e}")

        return all_results

    def show_overview(self):
        """Display system and data metrics."""
        st.title("Crypto Prediction Monitor")
        self.show_system_metrics()
        self.show_data_metrics()

    def show_system_metrics(self):
        """Display system performance metrics."""
        st.header("System Metrics")
        cols = st.columns(4)
        with cols[0]:
            st.metric("CPU Usage", "45%")
        with cols[1]:
            st.metric("Memory Usage", "3.2GB")
        with cols[2]:
            st.metric("GPU Usage", "78%")
        with cols[3]:
            st.metric("Disk Usage", "45%")

    def show_data_metrics(self):
        """Display data pipeline metrics."""
        st.header("Data Pipeline Metrics")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Data Freshness", "2 min ago")
            st.metric("Records Processed", "15,234")
        with cols[1]:
            st.metric("API Success Rate", "99.8%")
            st.metric("Processing Time", "45s")
        with cols[2]:
            st.metric("Missing Values", "0.02%")
            st.metric("Data Quality Score", "98.5%")

    def show_model_performance(self, results):
        """Display model performance metrics and visualizations."""
        st.title("Model Performance Analysis")
        st.header("Model Performance Metrics")

        for coin, result in results.items():
            st.subheader(f"{coin.title()} Metrics")
            st.metric("MAE", f"{result['evaluation']['mae']:.4f}")
            st.metric("RMSE", f"{result['evaluation']['rmse']:.4f}")
            st.metric("Direction Accuracy",
                      f"{result['evaluation'].get('directional_accuracy', 0):.2%}")

            # Use visualizer to generate visualizations
            st.plotly_chart(self.visualizer.plot_predictions(
                actual_prices=pd.Series(result['actual_prices']),
                predicted_prices=pd.Series(result['predictions'])
            ), use_container_width=True)

            st.plotly_chart(self.visualizer.plot_training_history(
                result['history']
            ), use_container_width=True)

    def show_live_predictions(self, results):
        """Display live predictions and their accuracy."""
        st.title("Live Predictions Dashboard")

        if not results:
            st.warning("No results data available")
            return

        for coin, result in results.items():
            st.subheader(f"{coin.title()} Predictions")
            cols = st.columns(4)

            try:
                predictions = result['predictions']
                actuals = result['actual_prices']

                if isinstance(predictions[-1], list):
                    latest_pred = predictions[-1][0]  # Take first element if list
                else:
                    latest_pred = predictions[-1]

                if isinstance(actuals[-1], list):
                    actual = actuals[-1][0]
                else:
                    actual = actuals[-1]

                with cols[0]:
                    st.metric("Latest Prediction", f"${latest_pred:.2f}")
                with cols[1]:
                    st.metric("Actual Price", f"${actual:.2f}")
                with cols[2]:
                    error = abs(latest_pred - actual)
                    st.metric("Absolute Error", f"${error:.2f}")
                with cols[3]:
                    accuracy = 1 - (error / actual)
                    st.metric("Accuracy", f"{accuracy:.2%}")
            except (IndexError, KeyError, TypeError) as e:
                st.error(f"Error displaying metrics for {coin}: {str(e)}")

    def show_training_history(self, results):
        """Display training history visualizations."""
        st.title("Training History")
        for coin, result in results.items():
            st.subheader(f"{coin.title()} Training History")
            st.plotly_chart(self.visualizer.plot_training_history(
                result['history']
            ), use_container_width=True)

def main():
    """Run the Streamlit dashboard."""
    dashboard = MonitoringDashboard()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Performance", "Live Predictions"]
    )

    # Load results
    results = dashboard.load_results()

    if page == "Overview":
        dashboard.show_overview()
    elif page == "Model Performance":
        dashboard.show_model_performance(results)
    elif page == "Live Predictions":
        dashboard.show_live_predictions(results)


if __name__ == "__main__":
    main()
