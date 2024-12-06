from unittest.mock import patch, AsyncMock

import pandas as pd
import pytest


@pytest.fixture
def mock_successful_response():
    """Mock successful API response"""
    return {
        'prices': [[1635724800000, 61500.0], [1635811200000, 63100.0]],
        'market_caps': [[1635724800000, 1160000000000], [1635811200000, 1190000000000]],
        'total_volumes': [[1635724800000, 28000000000], [1635811200000, 31000000000]]
    }


@pytest.fixture
def mock_binance_response():
    """Mock Binance API response"""
    return [
        [
            1635724800000,  # Open time
            "61500.0",  # Open
            "63100.0",  # High
            "61200.0",  # Low
            "62800.0",  # Close
            "1000.5",  # Volume
            1635811199999,  # Close time
            "62000000.0",  # Quote asset volume
            100,  # Number of trades
            "500.5",  # Taker buy base asset volume
            "31000000.0",  # Taker buy quote asset volume
            "0"  # Ignore
        ]
    ]


def test_data_collector_initialization(symbol_mapping):
    """Test DataCollector initialization"""
    from src.data_collection.data_collector import DataCollector

    collector = DataCollector(coins=['bitcoin'], days=365, symbol_mapping=symbol_mapping)
    assert collector.coins == ['bitcoin']
    assert collector.days == 365
    assert collector.coingecko_api == "https://api.coingecko.com/api/v3"
    assert collector.binance_api == "https://api.binance.com/api/v3"


def test_get_binance_symbol(symbol_mapping):
    """Test symbol matching functionality"""
    from src.data_collection.data_collector import DataCollector

    collector = DataCollector(coins=['bitcoin'], days=365, symbol_mapping=symbol_mapping)

    # Valid symbol matches
    assert collector.get_binance_symbol('BTC', 'USDT') == "BTCUSDT"
    assert collector.get_binance_symbol('ETH', 'BTC') == "ETHBTC"

    # Invalid symbol match
    assert collector.get_binance_symbol('DOGE', 'BTC') == ""  # Use a pair not in symbol_mapping




@pytest.mark.asyncio
async def test_fetch_coingecko_data_success(mock_successful_response, symbol_mapping):
    """Test successful data fetch from CoinGecko"""
    from src.data_collection.data_collector import DataCollector

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_successful_response
        mock_get.return_value.__aenter__.return_value = mock_response

        collector = DataCollector(coins=['bitcoin'], days=365, symbol_mapping=symbol_mapping)
        df = await collector.fetch_coingecko_data('bitcoin')

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ['price', 'market_cap', 'volume']
        assert len(df) == 2


@pytest.mark.asyncio
async def test_fetch_binance_data_success(mock_binance_response, symbol_mapping):
    """Test successful data fetch from Binance"""
    from src.data_collection.data_collector import DataCollector

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_binance_response
        mock_get.return_value.__aenter__.return_value = mock_response

        collector = DataCollector(coins=['bitcoin'], days=365, symbol_mapping=symbol_mapping)
        df = await collector.fetch_binance_data('BTC', 'USDT')

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert len(df) == 1


@pytest.mark.asyncio
async def test_fetch_binance_data_error_handling(symbol_mapping):
    """Test error handling in Binance data fetch"""
    from src.data_collection.data_collector import DataCollector

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.status = 404

        collector = DataCollector(coins=['bitcoin'], days=365, symbol_mapping=symbol_mapping)
        df = await collector.fetch_binance_data('BTC', 'USDT')

        assert df.empty  # Should return an empty DataFrame


@pytest.mark.asyncio
async def test_collect_all_data_with_symbol_mapping(mock_binance_response, mock_successful_response, symbol_mapping):
    """Test complete data collection process with symbol mapping"""
    from src.data_collection.data_collector import DataCollector

    with patch('src.data_collection.data_collector.DataCollector.fetch_coingecko_data') as mock_coingecko, \
         patch('src.data_collection.data_collector.DataCollector.fetch_binance_data') as mock_binance:

        mock_coingecko.return_value = pd.DataFrame({
            'price': [61500.0, 63100.0],
            'market_cap': [1160000000000, 1190000000000],
            'volume': [28000000000, 31000000000]
        }, index=pd.date_range(start='2021-01-01', periods=2))

        mock_binance.side_effect = lambda base, quote: (
            pd.DataFrame({
                'open': [61500.0],
                'high': [62000.0],
                'low': [61000.0],
                'close': [61800.0],
                'volume': [1000.5]
            }, index=pd.date_range(start='2021-01-01', periods=1))
            if collector.get_binance_symbol(base, quote)
            else pd.DataFrame()
        )

        collector = DataCollector(coins=['bitcoin'], days=365, symbol_mapping=symbol_mapping)
        data = await collector.collect_all_data()

        # Validate Binance data
        assert 'bitcoin' in data
        assert not data['bitcoin']['binance'].empty

        # Validate CoinGecko data
        assert not data['bitcoin']['coingecko'].empty
