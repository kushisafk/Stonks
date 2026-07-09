import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily returns, 5-day rolling returns, and 20-day rolling returns."""
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)
    return df

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 10-day, 20-day, and 50-day simple moving averages."""
    df = df.copy()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    return df

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 20-day and 50-day exponential moving averages."""
    df = df.copy()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df

def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 20-day rolling volatility of daily returns."""
    df = df.copy()
    if "daily_return" not in df.columns:
        df["daily_return"] = df["Close"].pct_change()
    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculates the Relative Strength Index (RSI) over a specified period (default 14)."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the MACD line (12-EMA - 26-EMA) and the 9-day MACD signal line."""
    df = df.copy()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df

def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Bollinger Bands: 20-day moving average +/- 2 standard deviations."""
    df = df.copy()
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["bb_middle"] = ma20
    df["bb_upper"] = ma20 + (2.0 * std20)
    df["bb_lower"] = ma20 - (2.0 * std20)
    return df
