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
            coins: List of cryptocurrency names.
            days: Number of days of historical data.
            symbol_mapping: List of dictionaries containing Binance symbol mappings.
            coin_map: Dictionary mapping coin names to base assets.
        """
        self.coins = coins
        self.days = days
        self.symbol_mapping = symbol_mapping
        self.coin_map = coin_map
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.binance_api = "https://api.binance.com/api/v3"

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not logging.getLogger(__name__).hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.validate_timeframe(days)

    def get_base_asset(self, coin: str) -> str:
        """
        Get the base asset for a given coin name.

        Args:
            coin: Coin name (e.g., 'ethereum', 'litecoin').

        Returns:
            Base asset string (e.g., 'ETH', 'LTC').
        """
        base_asset = self.coin_map.get(coin.lower())
        if base_asset is None:
            base_asset = coin[:3].upper()
            self.logger.warning(f"Base asset for {coin} not found in coin_map. Using default: {base_asset}")
        return base_asset

    def get_binance_symbol(self, base_asset: str, quote_asset: str) -> str:
        """
        Retrieve the Binance trading symbol for a given base and quote asset.
        """
        for entry in self.symbol_mapping:
            self.logger.debug(
                f"Checking symbol: {entry['symbol']}, Base: {entry['baseAsset']}, Quote: {entry['quoteAsset']}"
            )
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
    def validate_timeframe(days: int) -> bool:
        """
        Validate the timeframe for data collection.

        Args:
            days: Number of days to validate

        Returns:
            bool: True if timeframe is valid

        Raises:
            ValueError: If timeframe is invalid
        """
        if days <= 0:
            raise ValueError("Days must be positive")
        if days > 2000:
            raise ValueError("Maximum timeframe is 2000 days")
        return True

    async def fetch_coingecko_data(self, coin: str) -> pd.DataFrame:
        """
        Fetch historical data from CoinGecko API.

        Args:
            coin: Name of the cryptocurrency

        Returns:
            DataFrame with price, market cap, and volume data
        """
        try:
            endpoint = f"{self.coingecko_api}/coins/{coin}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': self.days,
                'interval': 'daily'
            }

            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"API request failed with status {response.status}")

                    data = await response.json()

                    df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                    df_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
                    df_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

                    df = df_price.merge(df_market_cap['market_cap'], left_index=True, right_index=True)
                    df = df.merge(df_volume['volume'], left_index=True, right_index=True)

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    self.logger.info(f"Successfully fetched CoinGecko data for {coin}")
                    return df

        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko data: {str(e)}")
            raise

    async def fetch_binance_data(self, base_asset: str, quote_asset: str) -> pd.DataFrame:
        """
        Fetch historical data from Binance API.

        Args:
            base_asset: The base asset (e.g., 'BTC').
            quote_asset: The quote asset (e.g., 'USDT').

        Returns:
            DataFrame with OHLCV data, or an empty DataFrame if an error occurs.
        """
        self.logger.debug(f"Fetching Binance data for base_asset={base_asset}, quote_asset={quote_asset}")
        try:
            symbol = self.get_binance_symbol(base_asset, quote_asset)
            if not symbol:
                self.logger.error(f"Symbol not found for {base_asset}/{quote_asset}")
                return pd.DataFrame()

            endpoint = f"{self.binance_api}/klines"
            params = {'symbol': symbol, 'interval': '1d', 'limit': self.days}

            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"API request failed with status {response.status}")

                    data = await response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignored'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df[['open', 'high', 'low', 'close', 'volume']] = df[
                        ['open', 'high', 'low', 'close', 'volume']].astype(float)
                    self.logger.info(f"Successfully fetched Binance data for {symbol}")
                    return df

        except Exception as e:
            self.logger.error(f"Error fetching Binance data for {base_asset}/{quote_asset}: {str(e)}")
            return pd.DataFrame()

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data by handling missing values and outliers.

        Args:
            df: Raw DataFrame to process

        Returns:
            Processed DataFrame
        """
        # Handle missing values
        if df.isnull().any().any():
            self.logger.warning("Found missing values in data")
            df = df.interpolate(method='time')  # Interpolate missing values
            df = df.bfill()  # Backfill any remaining NaNs


        return df

    async def collect_recent_data(self, coins: List[str], days_back: int, save_dir: Union[str, Path]) -> Dict[
        str, Dict[str, pd.DataFrame]]:
        self.logger.info(f"Collecting recent data for the last {days_back} days for prediction.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        all_data = {}

        for coin in coins:  # Only iterate over the filtered coins list
            try:
                base_asset = self.get_base_asset(coin)
                quote_asset = 'USDT'

                coingecko_data, binance_data = await asyncio.gather(
                    self.fetch_coingecko_data(coin),
                    self.fetch_binance_data(base_asset, quote_asset)
                )

                coingecko_data = self.process_raw_data(coingecko_data)
                binance_data = self.process_raw_data(binance_data)

                all_data[coin] = {
                    'coingecko': coingecko_data,
                    'binance': binance_data
                }

                for source, df in all_data[coin].items():
                    if not df.empty:
                        filename = f"{coin}_{source}_recent_{datetime.now().strftime('%Y%m%d')}.csv"
                        df.to_csv(save_dir / filename)
                        self.logger.info(f"Saved recent data for {coin} from {source} to {filename}")

            except Exception as e:
                self.logger.error(f"Error collecting recent data for {coin}: {str(e)}")

        self.logger.info("Finished collecting recent data for predictions.")
        return all_data

    async def collect_all_data(self, coins: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data from all sources for all specified coins.

        Returns:
            Dictionary containing DataFrames from each source for each coin.
        """
        coins_to_process = coins if coins else self.coins
        all_data = {}

        for coin in coins_to_process:
            try:
                base_asset = self.get_base_asset(coin)
                quote_asset = 'USDT'  # Default quote asset

                coingecko_data, binance_data = await asyncio.gather(
                    self.fetch_coingecko_data(coin),
                    self.fetch_binance_data(base_asset, quote_asset)
                )

                coingecko_data = self.process_raw_data(coingecko_data)
                binance_data = self.process_raw_data(binance_data)

                # Ensure structure is consistent
                all_data[coin] = {
                    'coingecko': coingecko_data,
                    'binance': binance_data
                }
                self.logger.info(f"Successfully collected all data for {coin}")

            except Exception as e:
                self.logger.error(f"Error collecting data for {coin}: {str(e)}")

        return all_data

    def save_data(self, data: Dict[str, Dict[str, pd.DataFrame]], save_dir: Union[str, Path]) -> None:
        """
        Save collected data to CSV files.

        Args:
            data: Dictionary containing the collected data
            save_dir: Directory to save the files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for coin, sources in data.items():
            for source, df in sources.items():
                self.logger.debug(f"Data for {coin} from {source}: {df.head()}")
                if df.empty or not isinstance(df, pd.DataFrame):
                    self.logger.error(f"Unexpected data format for {coin} from {source}")
                    continue
                filename = f"{coin}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = save_dir / filename
                df.to_csv(filepath)
                self.logger.info(f"Saved {filename}")


if __name__ == "__main__":
    import asyncio
    import logging
    import yaml
    from pathlib import Path
    import sys

    # Set logging to DEBUG level to see all details
    logging.getLogger().setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(stream_handler)

    # Load the same config.yaml as main.py would use
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("Error: configs/config.yaml not found.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Extract data from config
    data_config = config_data.get('data', {})
    coins = data_config.get('coins', [])
    days = data_config.get('days', 30)
    symbol_mapping = data_config.get('symbol_mapping', [])
    coin_map = data_config.get('coin_map', {})

    # Print out the extracted config to confirm correctness
    print("=== DEBUG CONFIG VALUES ===")
    print("Coins:", coins)
    print("Days:", days)
    print("Symbol Mapping:", symbol_mapping)
    print("Coin Map:", coin_map)
    print("===========================")

    if not coins:
        print("No coins found in config. Please add coins to config.yaml.")
        sys.exit(1)

    # Create DataCollector with the same parameters as main.py would
    collector = DataCollector(
        coins=coins,
        days=days,
        symbol_mapping=symbol_mapping,
        coin_map=coin_map
    )

    async def debug_collection():
        print("Starting debug data collection...")
        # We'll just collect the first coin to keep it simple
        test_coin = coins[0].lower()
        print(f"Attempting to collect data for coin: {test_coin}")

        try:
            # Try fetching data exactly as main.py's collect_data would
            data = await collector.collect_all_data([test_coin])
            print("Data collection attempt finished.")

            # Print out keys and check if coin data returned
            print("Collected data keys:", data.keys())
            if test_coin in data:
                coin_data = data[test_coin]
                print(f"Sources for '{test_coin}':", coin_data.keys())

                # Debug print a few rows from coingecko source
                coingecko_df = coin_data.get("coingecko", pd.DataFrame())
                print("CoinGecko DataFrame head:\n", coingecko_df.head())

                # Debug print a few rows from binance source
                binance_df = coin_data.get("binance", pd.DataFrame())
                print("Binance DataFrame head:\n", binance_df.head())
            else:
                print(f"No data returned for {test_coin}. Check logs for errors.")
        except Exception as e:
            print(f"An error occurred during data collection: {e}")

    asyncio.run(debug_collection())


