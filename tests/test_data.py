import os
import time
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data.market_data import MarketDataService
from src.data.news_data import MockNewsDataCollector

@patch("yfinance.Ticker")
def test_market_data_fetch_and_cache(mock_ticker, tmp_path):
    """Verify that market data is cached upon fetch, and subsequent requests hit the cache."""
    # Prepare dummy market data dataframe
    dates = pd.date_range(start="2026-05-01", periods=5, freq="D")
    dummy_data = {
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [105.0, 106.0, 107.0, 108.0, 109.0],
        "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "Close": [102.0, 103.0, 101.0, 104.0, 105.0],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    }
    dummy_df = pd.DataFrame(dummy_data, index=dates)
    dummy_df.index.name = "Date"
    
    # Configure yfinance mock instance
    mock_instance = MagicMock()
    mock_instance.history.return_value = dummy_df
    mock_ticker.return_value = mock_instance
    
    # Instantiate service with temporary cache directory
    service = MarketDataService(cache_dir=tmp_path, cache_expiry_seconds=30)
    
    # 1. Fetch data (Cache Miss -> Calls API)
    df1 = service.fetch_data("AAPL", period="1mo", interval="1d")
    assert len(df1) == 5
    assert list(df1.columns) == ["Open", "High", "Low", "Close", "Volume"]
    
    # Verify cache file was created
    expected_cache_path = tmp_path / "AAPL_1MO_1D.csv"
    assert expected_cache_path.exists()
    
    # 2. Modify cached file manually to prove subsequent calls hit cache
    cached_df = pd.read_csv(expected_cache_path, index_col="Date", parse_dates=["Date"])
    new_date = pd.to_datetime("2026-05-06")
    cached_df.loc[new_date] = [200.0, 205.0, 195.0, 202.0, 2000]
    cached_df.to_csv(expected_cache_path)
    
    # Fetch again without force (Cache Hit -> Returns modified data)
    df2 = service.fetch_data("AAPL", period="1mo", interval="1d")
    assert len(df2) == 6
    assert df2.index[-1] == new_date
    assert df2.iloc[-1]["Close"] == 202.0
    
    # 3. Fetch with force refresh (Bypasses cache -> Fetches original 5 mock rows)
    df3 = service.fetch_data("AAPL", period="1mo", interval="1d", force_refresh=True)
    assert len(df3) == 5
    assert df3.index[-1] != new_date
    
def test_news_mock_collector():
    """Verify that MockNewsDataCollector returns correctly structured fake news headlines."""
    collector = MockNewsDataCollector()
    news = collector.fetch_news("AAPL")
    
    assert len(news) > 0
    assert "headline" in news[0]
    assert "summary" in news[0]
    assert "published_at" in news[0]
    assert "source" in news[0]
    assert "AAPL" in news[0]["headline"]
