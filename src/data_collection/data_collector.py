# src/data_collection/data_collector.py
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import aiohttp
import pandas as pd


class DataCollector:
    def __init__(self, coins: List[str], days: int, symbol_mapping: List[Dict[str, str]], coin_map: Dict[str, str]):
        """
        Initialize the DataCollector.

        Args:
            coins (List[str]): List of cryptocurrency names.
            days (int): Number of days of historical data to collect.
            symbol_mapping (List[Dict[str, str]]): Binance symbol mappings.
            coin_map (Dict[str, str]): Mapping of coin names to their base assets.
        """
        self.coins = coins
        self.days = days
        self.symbol_mapping = symbol_mapping
        self.coin_map = coin_map
        self.binance_api = "https://api.binance.com/api/v3"

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logger()

        # Validate the timeframe for historical data
        self.validate_timeframe(days)

    def _setup_logger(self):
        """Set up logger with a consistent configuration."""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_base_asset(self, coin: str) -> str:
        """
        Retrieve the base asset for a given coin name.

        Args:
            coin (str): Coin name (e.g., 'ethereum', 'litecoin').

        Returns:
            str: Base asset symbol (e.g., 'ETH', 'LTC').
        """
        base_asset = self.coin_map.get(coin.lower())
        if base_asset is None:
            base_asset = coin[:3].upper()
            self.logger.warning(f"Base asset for {coin} not found in coin_map. Using default: {base_asset}")
        return base_asset

    def get_binance_symbol(self, base_asset: str, quote_asset: str) -> str:
        """
        Retrieve the Binance trading symbol for a given base and quote asset.

        Args:
            base_asset (str): Base asset symbol (e.g., 'BTC').
            quote_asset (str): Quote asset symbol (e.g., 'USDT').

        Returns:
            str: Binance trading symbol (e.g., 'BTCUSDT'), or an empty string if not found.
        """
        for entry in self.symbol_mapping:
            if entry['baseAsset'] == base_asset and entry['quoteAsset'] == quote_asset:
                return entry['symbol']

        warning_message = f"No Binance symbol found for {base_asset}/{quote_asset}"
        if not hasattr(self, '_logged_warnings'):
            self._logged_warnings = set()
        if warning_message not in self._logged_warnings:
            self.logger.warning(warning_message)
            self._logged_warnings.add(warning_message)

        return ""

    @staticmethod
    def validate_timeframe(days: int) -> None:
        """
        Validate the timeframe for data collection.

        Args:
            days (int): Number of days of historical data.

        Raises:
            ValueError: If the timeframe is invalid.
        """
        if days <= 0:
            raise ValueError("Days must be a positive integer.")
        if days > 2000:
            raise ValueError("The maximum allowable timeframe is 2000 days.")

    async def fetch_binance_data(self, base_asset: str, quote_asset: str) -> pd.DataFrame:
        """
        Fetch historical data from the Binance API.

        Args:
            base_asset (str): Base asset (e.g., 'BTC').
            quote_asset (str): Quote asset (e.g., 'USDT').

        Returns:
            pd.DataFrame: DataFrame with OHLCV data, or an empty DataFrame on failure.
        """
        symbol = self.get_binance_symbol(base_asset, quote_asset)
        if not symbol:
            self.logger.warning(f"Binance symbol not found for {base_asset}/{quote_asset}")
            return pd.DataFrame()

        endpoint = f"{self.binance_api}/klines"
        params = {'symbol': symbol, 'interval': '1d', 'limit': self.days}

        self.logger.info(f"Fetching Binance data for symbol={symbol} with days={self.days}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"Binance API request failed with status {response.status}")

                    data = await response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignored'
                    ])

                    # Format and process DataFrame
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                    self.logger.info(f"Successfully fetched Binance data for {symbol}")
                    return df

        except Exception as e:
            self.logger.error(f"Error fetching Binance data for {base_asset}/{quote_asset}: {str(e)}")
            return pd.DataFrame()

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data by handling missing values and basic data cleaning.

        Args:
            df (pd.DataFrame): Raw DataFrame to process.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        if df.empty:
            self.logger.warning("Received an empty DataFrame for processing")
            return df

        if df.isnull().any().any():
            self.logger.warning("Found missing values in the data. Interpolating and backfilling.")
            df = df.interpolate(method='time')
            df = df.bfill()

        return df

    async def collect_all_data(self, coins: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data from Binance for the specified coins.

        Args:
            coins (List[str]): List of coin names. If None, use the object's configured coins.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Data dictionary with binance data for each coin.
        """
        coins_to_process = coins if coins else self.coins
        self.logger.info("Starting collection of all data for specified coins.")
        all_data = {}

        for coin in coins_to_process:
            try:
                base_asset = self.get_base_asset(coin)
                quote_asset = 'USDT'

                binance_data = await self.fetch_binance_data(base_asset, quote_asset)
                binance_data = self.process_raw_data(binance_data)

                all_data[coin] = {
                    'binance': binance_data
                }
                self.logger.info(f"Successfully collected data for {coin}.")

            except Exception as e:
                self.logger.error(f"Error collecting data for {coin}: {str(e)}")

        self.logger.info("Completed collection of all data.")
        return all_data

    def save_data(self, data: Dict[str, Dict[str, pd.DataFrame]], save_dir: Union[str, Path]) -> None:
        """
        Save collected data to CSV files.

        Args:
            data (dict): Dictionary containing the collected data.
            save_dir (Union[str, Path]): Directory to save the files.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving collected data to directory: {save_dir}")

        for coin, sources in data.items():
            for source, df in sources.items():
                if df.empty or not isinstance(df, pd.DataFrame):
                    self.logger.warning(f"Data for {coin} from {source} is empty or invalid. Skipping save.")
                    continue
                filename = f"{coin}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = save_dir / filename
                try:
                    df.to_csv(filepath)
                    self.logger.info(f"Saved data for {coin} from {source} to {filename}.")
                except Exception as e:
                    self.logger.error(f"Failed to save data for {coin} from {source}: {str(e)}")


if __name__ == "__main__":
    import asyncio

    # Example configuration
    test_coins = ["bitcoin"]
    test_days = 30
    test_symbol_mapping = [
        {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT"
        }
    ]
    test_coin_map = {"bitcoin": "BTC"}

    # Initialize DataCollector
    collector = DataCollector(
        coins=test_coins,
        days=test_days,
        symbol_mapping=test_symbol_mapping,
        coin_map=test_coin_map
    )

    # Specify save directory
    save_dir = "data/raw/train"

    # Run data collection
    data = asyncio.run(collector.collect_all_data())
    collector.save_data(data, save_dir)
