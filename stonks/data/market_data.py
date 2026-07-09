import os
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional
from stonks.config.settings import settings
from stonks.logging.logger import logger

class MarketDataService:
    """Fetches market data from yfinance and manages a local file-based cache to avoid redundant API hits."""
    
    def __init__(self, cache_dir: Optional[Path] = None, cache_expiry_seconds: int = 14400):
        """
        Args:
            cache_dir: Directory to save raw stock data. Defaults to settings.CACHE_DIR.
            cache_expiry_seconds: Time in seconds before cache is considered expired (default 4 hours).
        """
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_seconds = cache_expiry_seconds
        
    def _get_cache_path(self, symbol: str, period: str, interval: str) -> Path:
        """Constructs a unique cache file path."""
        clean_symbol = symbol.strip().upper()
        return self.cache_dir / f"{clean_symbol}_{period}_{interval}.csv"
        
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Checks if cached file exists and is not expired."""
        if not cache_path.exists():
            return False
            
        mtime = cache_path.stat().st_mtime
        age = time.time() - mtime
        return age < self.cache_expiry_seconds
        
    def fetch_data(
        self, 
        symbol: str, 
        period: Optional[str] = None, 
        interval: Optional[str] = None, 
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetches historical market data for a symbol. Tries local cache first.
        
        Args:
            symbol: Ticker symbol (e.g., AAPL).
            period: History window (e.g., 2y). Defaults to settings.YFINANCE_PERIOD.
            interval: Standard interval (e.g., 1d). Defaults to settings.YFINANCE_INTERVAL.
            force_refresh: If True, bypass cache and fetch directly from yfinance.
            
        Returns:
            pd.DataFrame: Contains historical data with standard columns: 
                          Open, High, Low, Close, Volume, and a Date column/index.
        """
        symbol = symbol.strip().upper()
        period = period or settings.YFINANCE_PERIOD
        interval = interval or settings.YFINANCE_INTERVAL
        
        cache_path = self._get_cache_path(symbol, period, interval)
        
        if not force_refresh and self._is_cache_valid(cache_path):
            try:
                logger.info(f"Cache hit: Loading data for {symbol} ({period}, {interval}) from {cache_path}")
                df = pd.read_csv(cache_path, parse_dates=["Date"])
                if not df.empty:
                    df.set_index("Date", inplace=True)
                    # Clean up DatetimeIndex naming/type consistency
                    df.index = pd.to_datetime(df.index)
                    return df
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}, falling back to API: {e}")
                
        # Cache miss or expired, fetch from yfinance
        logger.info(f"Cache miss/refresh: Fetching data for {symbol} ({period}, {interval}) from yfinance")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No market data returned from yfinance for {symbol}")
                
            # Clean up the index and columns
            df.index = pd.to_datetime(df.index)
            # Ensure the DatetimeIndex is timezone naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            # Rename index to 'Date' for clean persistence
            df.index.name = "Date"
            
            # Select and order standard required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    raise KeyError(f"Required column '{col}' is missing in yfinance response.")
                    
            df = df[required_cols]
            
            # Save to cache
            df.to_csv(cache_path)
            logger.info(f"Successfully cached market data to {cache_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {symbol}: {e}")
            # If API fetch fails but expired cache exists, return expired cache as emergency fallback
            if cache_path.exists():
                logger.warning(f"Emergency recovery: returning expired cache data for {symbol}")
                try:
                    df = pd.read_csv(cache_path, parse_dates=["Date"])
                    df.set_index("Date", inplace=True)
                    df.index = pd.to_datetime(df.index)
                    return df
                except Exception as cache_err:
                    logger.error(f"Failed to read expired cache during recovery: {cache_err}")
            raise e

# Global market data service instance
market_data_service = MarketDataService()
