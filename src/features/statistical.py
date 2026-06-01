import pandas as pd
import numpy as np

def add_rolling_skew(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates the rolling skewness of daily returns over the specified window."""
    df = df.copy()
    if "daily_return" not in df.columns:
        df["daily_return"] = df["Close"].pct_change()
    df["skew_20d"] = df["daily_return"].rolling(window).skew()
    return df

def add_rolling_kurt(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates the rolling kurtosis of daily returns over the specified window."""
    df = df.copy()
    if "daily_return" not in df.columns:
        df["daily_return"] = df["Close"].pct_change()
    df["kurt_20d"] = df["daily_return"].rolling(window).kurt()
    return df
