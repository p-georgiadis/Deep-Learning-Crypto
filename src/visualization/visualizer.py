# src/visualization/visualizer.py

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CryptoVisualizer:
    """
    A simplified visualizer for cryptocurrency price prediction tasks.

    Provides methods for:
    - Plotting price history (candlestick + volume)
    - Plotting model performance metrics
    - Plotting actual vs. predicted prices
    - Plotting training history (loss/metrics over epochs)
    - Plotting correlation matrix
    """

    def __init__(self, style: str = 'plotly_white', figure_size: Tuple[int, int] = (1000, 600)):
        self.style = style
        self.figure_size = figure_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def plot_price_history(self, df: pd.DataFrame) -> go.Figure:
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError("DataFrame must contain open, high, low, close, volume columns.")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume'
        ), row=2, col=1)

        fig.update_layout(
            title='Price History',
            template=self.style,
            height=self.figure_size[1],
            width=self.figure_size[0],
            xaxis_rangeslider_visible=False
        )
        return fig

    def plot_performance_metrics(self, metrics: Dict[str, float]) -> go.Figure:
        if not metrics:
            raise ValueError("Metrics dictionary cannot be empty.")

        fig = go.Figure(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=[f"{v:.4f}" for v in metrics.values()],
            textposition='auto'
        ))
        fig.update_layout(
            title='Model Performance Metrics',
            template=self.style,
            height=self.figure_size[1],
            width=self.figure_size[0]
        )
        return fig

    def plot_predictions(self, actual_prices: pd.Series, predicted_prices: pd.Series) -> go.Figure:
        if len(actual_prices) != len(predicted_prices):
            raise ValueError("Actual and predicted price series must have the same length.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual_prices.index,
            y=actual_prices.values,
            name='Actual',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=predicted_prices.index,
            y=predicted_prices.values,
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='Actual vs Predicted Prices',
            template=self.style,
            height=self.figure_size[1],
            width=self.figure_size[0]
        )
        return fig

    def plot_training_history(self, history: Dict[str, List[float]]) -> go.Figure:
        if not history:
            raise ValueError("History dictionary cannot be empty.")

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Loss', 'Metrics'),
                            vertical_spacing=0.1)

        # Plot loss
        if 'loss' in history:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['loss']) + 1)),
                y=history['loss'],
                name='Train Loss',
                line=dict(color='blue')
            ), row=1, col=1)

        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history['val_loss']) + 1)),
                y=history['val_loss'],
                name='Val Loss',
                line=dict(color='red')
            ), row=1, col=1)

        # Plot other metrics (except loss/val_loss)
        metric_keys = [k for k in history.keys() if k not in ('loss', 'val_loss')]
        for m in metric_keys:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history[m]) + 1)),
                y=history[m],
                name=m.capitalize(),
            ), row=2, col=1)

        fig.update_layout(
            title='Training History',
            template=self.style,
            height=self.figure_size[1],
            width=self.figure_size[0]
        )
        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")

        corr = df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            hoverongaps=False
        ))
        fig.update_layout(
            title='Correlation Matrix',
            template=self.style,
            height=self.figure_size[1],
            width=self.figure_size[0]
        )
        return fig

    def save_plot(self, fig: go.Figure, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        self.logger.info(f"Plot saved to {path}")
