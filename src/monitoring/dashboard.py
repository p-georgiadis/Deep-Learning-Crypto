# src/monitoring/dashboard.py

import streamlit as st
import pandas as pd
import json
import logging
from pathlib import Path
from src.visualization.visualizer import CryptoVisualizer

class MonitoringDashboard:
    """
    A simplified Streamlit dashboard that:
    - Shows a title and model performance metrics
    - Displays a single plot for demonstration
    """

    def __init__(self, results_path: str = "results/results.json"):
        st.set_page_config(page_title="Crypto Prediction Monitor", page_icon="📈", layout="wide")
        self.logger = self._setup_logger()
        self.visualizer = CryptoVisualizer()
        self.results_path = results_path
        self.results = self._load_results()

    def _setup_logger(self):
        logger = logging.getLogger("MonitoringDashboard")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_results(self):
        path = Path(self.results_path)
        if not path.exists():
            self.logger.warning("Results file not found.")
            return {}
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return {}

    def show(self):
        st.title("Crypto Prediction Dashboard")
        if not self.results:
            st.warning("No results found.")
            return

        # Assume results has keys: 'evaluation' for metrics and 'history' for training history
        evaluation = self.results.get('evaluation', {})
        if evaluation:
            st.subheader("Performance Metrics")
            for metric, value in evaluation.items():
                st.metric(metric.capitalize(), f"{value:.4f}")

            # Demonstrate plotting training history if available
            history = self.results.get('history', {})
            if history:
                fig = self.visualizer.plot_training_history(history)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No evaluation metrics found in results.")


def main():
    dashboard = MonitoringDashboard()
    dashboard.show()

if __name__ == "__main__":
    main()
