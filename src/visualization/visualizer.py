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
    Creates visualizations for cryptocurrency data and model results.

    This class handles:
    - Price history visualization
    - Technical indicators plotting
    - Model prediction visualization
    - Training history plots
    - Performance metrics visualization
    """

    def __init__(self, style: str = 'plotly_dark', figure_size: Tuple[int, int] = (1200, 800)):
        """
        Initialize the visualizer.

        Args:
            style: Plotly template style.
            figure_size: Default figure size (width, height).
        """
        self.style = style
        self.figure_size = figure_size

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

    def plot_price_history(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a price history visualization with volume.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.

        Returns:
            A Plotly figure object.
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if df.empty or not all(col in df.columns for col in required_columns):
                raise ValueError("Input DataFrame must contain OHLCV data.")

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume'
                ),
                row=2, col=1
            )

            fig.update_layout(
                title='Cryptocurrency Price History',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0],
                xaxis_rangeslider_visible=False
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting price history: {str(e)}")
            raise

    def plot_performance_metrics(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Plot performance metrics as a bar chart.

        Args:
            metrics: Dictionary of performance metrics (e.g., MAE, RMSE).

        Returns:
            A Plotly figure object.
        """
        try:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=list(metrics.values()),
                textposition='auto',
                name='Performance Metrics'
            ))

            fig.update_layout(
                title='Model Performance Metrics',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0],
                xaxis_title='Metrics',
                yaxis_title='Values'
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}")
            raise

    def plot_technical_indicators(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot technical indicators (e.g., SMA, RSI, MACD).

        Args:
            df: DataFrame with price and technical indicators.

        Returns:
            A Plotly figure object.
        """
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price and Moving Averages', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )

            # Price and MA
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            for ma in ['sma_20', 'sma_50']:
                if ma in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[ma],
                            name=ma.upper(),
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )

            # RSI
            if 'rsi' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['rsi'],
                        name='RSI',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            if all(x in df.columns for x in ['macd', 'macd_signal']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd'],
                        name='MACD',
                        line=dict(width=2)
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd_signal'],
                        name='Signal',
                        line=dict(width=2)
                    ),
                    row=3, col=1
                )

            fig.update_layout(
                title='Technical Analysis Dashboard',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0],
                showlegend=True
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting technical indicators: {str(e)}")
            raise

    def plot_predictions(self, actual_prices: pd.Series, predicted_prices: pd.Series) -> go.Figure:
        """
        Plot actual vs. predicted prices.

        Args:
            actual_prices: Actual price series.
            predicted_prices: Predicted price series.

        Returns:
            A Plotly figure object.
        """
        try:
            if len(actual_prices) != len(predicted_prices):
                raise ValueError("Lengths of actual and predicted prices must match")
            if not isinstance(actual_prices, pd.Series) or not isinstance(predicted_prices, pd.Series):
                raise ValueError("Both actual and predicted prices must be pandas.Series")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=actual_prices.index,
                    y=actual_prices,
                    name='Actual',
                    line=dict(color='blue', width=2)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=predicted_prices.index,
                    y=predicted_prices,
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                )
            )

            fig.update_layout(
                title='Actual vs Predicted Prices',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0],
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting predictions: {str(e)}")
            raise

    def plot_training_history(self, history: Dict[str, List[float]]) -> go.Figure:
        """
        Plot training history metrics (loss and other metrics).

        Args:
            history: Training history dictionary from model.fit()

        Returns:
            A Plotly figure object.
        """
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Loss', 'Metrics'),
                vertical_spacing=0.1
            )

            # Plot loss
            fig.add_trace(
                go.Scatter(
                    y=history['loss'],
                    name='Training Loss',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(
                        y=history['val_loss'],
                        name='Validation Loss',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )

            # Plot other metrics
            metrics = [key for key in history.keys() if key not in ['loss', 'val_loss', 'lr']]

            for metric in metrics:
                fig.add_trace(
                    go.Scatter(
                        y=history[metric],
                        name=metric,
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )

            fig.update_layout(
                title='Training History',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0],
                showlegend=True
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")
            raise

    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot a correlation matrix heatmap.

        Args:
            df: DataFrame containing features to correlate.

        Returns:
            A Plotly figure object.
        """
        try:
            corr_matrix = df.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))

            fig.update_layout(
                title='Feature Correlation Matrix',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0]
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {str(e)}")
            raise

    def save_plot(self, fig: go.Figure, path: Union[str, Path]) -> None:
        """
        Save a plot to an HTML file.

        Args:
            fig: Plotly figure
            path: Save path
        """
        try:
            fig.write_html(str(path))
            self.logger.info(f"Plot saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving plot: {str(e)}")
            raise

    def plot_feature_importance(self, features: List[str], importance_scores: List[float]) -> go.Figure:
        """
        Plot feature importance as a bar chart.

        Args:
            features: List of feature names.
            importance_scores: Corresponding importance scores.

        Returns:
            A Plotly figure object.
        """
        try:
            fig = go.Figure(
                data=[go.Bar(x=features, y=importance_scores, text=importance_scores,
                             textposition='auto', name='Feature Importance')]
            )

            fig.update_layout(
                title='Feature Importance',
                template=self.style,
                height=self.figure_size[1],
                width=self.figure_size[0],
                xaxis_title='Features',
                yaxis_title='Importance',
                showlegend=False
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage (for quick testing)
    # You can remove or keep this block as needed.
    import yfinance as yf

    # Download sample data
    btc = yf.download('BTC-USD', start='2020-01-01', end='2020-01-10')  # short period for testing

    visualizer = CryptoVisualizer()
    fig = visualizer.plot_price_history(btc)
    visualizer.save_plot(fig, "price_history.html")
