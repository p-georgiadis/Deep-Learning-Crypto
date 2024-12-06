# src/preprocessing/feature_engineering.py
import logging
from typing import Tuple

import pandas as pd


class FeatureEngineer:
    """
    Handles feature engineering for cryptocurrency data.

    This class creates technical indicators and additional features
    for cryptocurrency price prediction.
    """

    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            prices: Series of prices
            periods: RSI period (default: 14)

        Returns:
            Series containing RSI values
        """
        try:
            delta = prices.diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Calculate average gains and losses
            avg_gains = gains.rolling(window=periods, min_periods=1).mean()
            avg_losses = losses.rolling(window=periods, min_periods=1).mean()

            # Calculate RS and RSI
            epsilon = 1e-10
            rs = avg_gains / (avg_losses + epsilon)
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def calculate_macd(self, prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            prices: Series of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, MACD histogram)
        """
        try:
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate Signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # Calculate MACD histogram
            macd_hist = macd_line - signal_line

            return macd_line, signal_line, macd_hist

        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def calculate_bollinger_bands(self, prices: pd.Series,
                                  window: int = 20,
                                  num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of prices
            window: Rolling window period
            num_std: Number of standard deviations

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        try:
            middle_band = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()

            upper_band = middle_band + (rolling_std * num_std)
            lower_band = middle_band - (rolling_std * num_std)

            return upper_band, middle_band, lower_band

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def calculate_moving_averages(self, prices: pd.Series, periods: Tuple[int, int] = (20, 50)) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Simple Moving Averages.

        Args:
            prices: Series of prices

        Returns:
            Tuple of (20-day SMA, 50-day SMA)
        """
        try:
            sma_1 = prices.rolling(window=periods[0]).mean()
            sma_2 = prices.rolling(window=periods[1]).mean()
            return sma_1, sma_2

            return sma_20, sma_50

        except Exception as e:
            self.logger.error(f"Error calculating Moving Averages: {str(e)}")
            raise

    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate price volatility.

        Args:
            prices: Series of prices
            window: Rolling window period

        Returns:
            Series containing volatility values
        """
        try:
            return prices.rolling(window=window).std()

        except Exception as e:
            self.logger.error(f"Error calculating Volatility: {str(e)}")
            raise

    def calculate_momentum_indicators(self, prices: pd.Series, period: int = 12) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate momentum indicators.

        Args:
            prices: Series of prices

        Returns:
            Tuple of (Rate of Change, Momentum)
        """
        try:
            roc = prices.pct_change(periods=period) * 100
            momentum = prices - prices.shift(period)
            return roc, momentum

        except Exception as e:
            self.logger.error(f"Error calculating Momentum Indicators: {str(e)}")
            raise

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with additional technical indicators.
        """
        try:
            if df.empty:
                raise ValueError("Empty DataFrame provided")

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Required: {required_columns}")
            if df[required_columns].isnull().any().any():
                raise ValueError("Missing values detected in critical columns")

            # Create a copy to avoid modifying the original data
            df_features = df.copy()

            # Calculate RSI
            df_features['rsi'] = self.calculate_rsi(df_features['close'])

            # Calculate MACD
            macd, signal, hist = self.calculate_macd(df_features['close'])
            df_features['macd'] = macd
            df_features['macd_signal'] = signal
            df_features['macd_hist'] = hist

            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df_features['close'])
            df_features['bb_upper'] = bb_upper
            df_features['bb_middle'] = bb_middle
            df_features['bb_lower'] = bb_lower

            # Calculate Moving Averages
            sma_20, sma_50 = self.calculate_moving_averages(df_features['close'])
            df_features['sma_20'] = sma_20
            df_features['sma_50'] = sma_50

            # Calculate Volatility
            df_features['volatility'] = self.calculate_volatility(df_features['close'])

            # Calculate daily returns
            df_features['daily_return'] = df_features['close'].pct_change()

            # Calculate momentum indicators
            roc, momentum = self.calculate_momentum_indicators(df_features['close'])
            df_features['roc'] = roc
            df_features['momentum'] = momentum

            if df_features.isnull().any().any():
                self.logger.warning("NaNs detected after feature engineering. Filling NaNs with forward fill.")
                df_features = df_features.ffill()
                df_features.dropna(inplace=True)  # Ensure no remaining NaNs

            self.logger.info(f"Data length after feature engineering and dropping NaNs: {len(df_features)}")
            self.logger.info("Successfully added all technical features.")
            return df_features

        except Exception as e:
            self.logger.error(f"Error adding technical features: {str(e)}")
            raise

    def create_target_variable(self, df: pd.DataFrame,
                               forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create target variable for model training.

        Args:
            df: DataFrame with features
            forecast_horizon: Number of periods ahead to predict

        Returns:
            Tuple of (Feature DataFrame, Target Series)
        """
        try:
            # Create future price column
            target = df['close'].shift(-forecast_horizon)

            # Create return-based target (percentage change)
            price_change = (target - df['close']) / df['close']

            # Remove last rows where target is NaN
            df = df[:-forecast_horizon]
            price_change = price_change[:-forecast_horizon]

            return df, price_change

        except Exception as e:
            self.logger.error(f"Error creating target variable: {str(e)}")
            raise
